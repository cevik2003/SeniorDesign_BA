"""
Beam Alignment - Few-Shot Adaptation: 5-Method Comparison
=========================================================
Pipeline:
  Phase 1 - Pre-train  : TinyCNN on BS3..BS10, checkpoint by BS11/12/13 val
  Phase 2 - Meta-train : MTL-style sequential SS + classifier carry-over
  Phase 2b- Meta-train : MAML from scratch (full model, episodic meta-learning)
  Phase 3 - Sweep      : Compare 5 methods over shots_list on target BS

LEAKAGE CONTROL:
  - Pre-train checkpoint selection uses only BS11/12/13 val (target BS never seen)
  - Meta-train uses only train_bss
  - Adaptation support and test never mix
  - No early stopping on target test
  - Target test is only evaluated after adaptation
"""

import time
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# =====================
# REPRODUCIBILITY
# =====================
rng = np.random.default_rng(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# =====================
# SETTINGS
# =====================
train_bss        = list(range(3, 11))   # BS3..BS10
val_bss          = [11, 12, 13]         # pre-train val only
target_bs        = 14                   # adaptation + test only

cap_per_bs_train = 40_000
cap_per_bs_val   = 5_000
cap_target_pool  = 40_000

epochs_pre       = 40
shots_list       = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

batch_train      = 2048
batch_test       = 2048
batch_ft         = 64

lr_pre           = 1e-3
wd               = 1e-4
label_smoothing  = 0.05

lr_ft_all        = 1e-4
lr_ft_last       = 3e-4
lr_ss_adapt      = 1e-3

# Meta-train (MTL-style) — unchanged
meta_epochs        = 1000
episodes_per_epoch = 30
n_shot_meta        = 20
n_query_meta       = 50
inner_steps        = 5
inner_lr           = 0.1
meta_lr            = 1e-3

# Optional meta-val for checkpointing (still leakage-safe since target is excluded)
use_meta_val       = True
meta_val_episodes  = 30

# MAML settings — ADDED
maml_epochs       = 1000
maml_episodes     = 30
maml_inner_steps  = 5
maml_inner_lr     = 1e-2
maml_meta_lr      = 1e-3
maml_adapt_lr     = 1e-2
maml_adapt_steps  = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# =====================
# HELPERS
# =====================
def load_bs_capped(bs, cap, top_by_max=False):
    mat = loadmat(
        f"beam_dataset_bs{bs}.mat",
        variable_names=["X", "best128"],
        verify_compressed_data_integrity=False
    )
    X = mat["X"].astype(np.float32)
    y = mat["best128"].reshape(-1).astype(np.int64)

    if y.min() == 1 and y.max() == 128:
        y = y - 1

    if cap is not None and len(X) > cap:
        if top_by_max:
            scores = X.max(axis=1)                 # each user in R^32 -> scalar score
            idx = np.argpartition(scores, -cap)[-cap:]   # top-cap unsorted
            idx = idx[np.argsort(scores[idx])[::-1]]     # optional: descending order
        else:
            idx = rng.choice(len(X), cap, replace=False)

        X, y = X[idx], y[idx]

    return X, y

def ft_epochs(shots, mode):
    if shots <= 10:  return 50
    if shots <= 50:  return 35
    if shots <= 200: return 25
    if shots <= 500: return 18
    return 12

def ss_adapt_epochs(shots):
    if shots <= 10:  return 50
    if shots <= 50:  return 35
    if shots <= 200: return 25
    if shots <= 500: return 18
    return 12

@torch.no_grad()
def evaluate(model, loader, topk=(1, 2, 3, 5)):
    model.eval()
    total = 0
    correct_k = {k: 0 for k in topk}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1)
        pred = pred.t()
        matches = pred.eq(yb.view(1, -1))
        for k in topk:
            correct_k[k] += matches[:k].any(dim=0).sum().item()
        total += xb.size(0)
    return {k: correct_k[k] / total for k in topk}

def zero_module_grads(module):
    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

# =====================
# MODELS
# =====================
class TinyCNN(nn.Module):
    def __init__(self, in_features=32, num_classes=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * in_features, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(1)
        return self.head(x)

    # ===== ADDED FOR MAML =====
    def forward_with_fast_weights(self, x, fast_weights):
        x = x.unsqueeze(1)

        x = F.conv1d(
            x,
            fast_weights['cnn.0.weight'],
            fast_weights['cnn.0.bias'],
            padding=1
        )
        x = F.relu(x)

        x = F.conv1d(
            x,
            fast_weights['cnn.2.weight'],
            fast_weights['cnn.2.bias'],
            padding=1
        )
        x = F.relu(x)

        x = x.flatten(1)

        x = F.linear(
            x,
            fast_weights['head.0.weight'],
            fast_weights['head.0.bias']
        )
        x = F.relu(x)

        x = F.linear(
            x,
            fast_weights['head.2.weight'],
            fast_weights['head.2.bias']
        )
        return x

    def get_fast_weights(self):
        return OrderedDict((name, p) for name, p in self.named_parameters())


class ScaleShift(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta  = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return self.gamma * x + self.beta

class TinyCNNWithSS(nn.Module):
    def __init__(self, in_features=32, num_classes=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.fc1   = nn.Linear(64 * in_features, 256)
        self.fc2   = nn.Linear(256, num_classes)

        self.ss1 = ScaleShift((1, 64, 1))
        self.ss2 = ScaleShift((1, 64, 1))
        self.ss3 = ScaleShift((1, 256))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.ss1(F.relu(self.conv1(x)))
        x = self.ss2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.ss3(F.relu(self.fc1(x)))
        return self.fc2(x)

    def ss_params(self):
        return [*self.ss1.parameters(), *self.ss2.parameters(), *self.ss3.parameters()]

    def head_params(self):
        return list(self.fc2.parameters())

    def backbone_params(self):
        return [*self.conv1.parameters(), *self.conv2.parameters(), *self.fc1.parameters()]

    def freeze_backbone(self):
        for p in self.backbone_params():
            p.requires_grad = False
        for p in self.ss_params() + self.head_params():
            p.requires_grad = True

    def reset_ss(self):
        for ss in [self.ss1, self.ss2, self.ss3]:
            nn.init.ones_(ss.gamma)
            nn.init.zeros_(ss.beta)

def load_tinycnn_weights_into_ss(ss_model, tiny_state_dict):
    mapping = {
        'cnn.0.weight': 'conv1.weight', 'cnn.0.bias': 'conv1.bias',
        'cnn.2.weight': 'conv2.weight', 'cnn.2.bias': 'conv2.bias',
        'head.0.weight': 'fc1.weight',  'head.0.bias': 'fc1.bias',
        'head.2.weight': 'fc2.weight',  'head.2.bias': 'fc2.bias',
    }
    new_state = ss_model.state_dict()
    for old_key, new_key in mapping.items():
        if old_key in tiny_state_dict:
            new_state[new_key] = tiny_state_dict[old_key]
        else:
            print(f"[WARN] missing key: {old_key}")
    ss_model.load_state_dict(new_state)

# =====================
# DATA LOADING
# =====================
print("Loading data...")

# BS3..BS13: top-50k by max(X_i)

# train BS
Xs_list, ys_list = [], []
for bs in train_bss:
    X, y = load_bs_capped(bs, 50_000, top_by_max=True)
    Xs_list.append(X)
    ys_list.append(y)

Xtr = np.vstack(Xs_list)
ytr = np.concatenate(ys_list)

mu  = Xtr.mean(0, keepdims=True)
std = Xtr.std(0, keepdims=True)

Xtr     = (Xtr - mu) / std
Xs_norm = [(X - mu) / std for X in Xs_list]

# val BS
Xval_list, yval_list = [], []
for bs in val_bss:
    X, y = load_bs_capped(bs, 50_000, top_by_max=True)
    Xval_list.append((X - mu) / std)
    yval_list.append(y)

Xval = np.vstack(Xval_list)
yval = np.concatenate(yval_list)

# target BS14 unchanged
X_target, y_target = load_bs_capped(target_bs, cap_target_pool, top_by_max=False)
X_target = (X_target - mu) / std

# target BS14 unchanged
X_target, y_target = load_bs_capped(target_bs, cap_target_pool, top_by_max=False)
X_target = (X_target - mu) / std

print(f"Train: {Xtr.shape} | Val: {Xval.shape} | Target BS{target_bs}: {X_target.shape}")

pin = (device.type == "cuda")

train_loader = DataLoader(
    TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
    batch_size=batch_train, shuffle=True, pin_memory=pin, num_workers=0
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(Xval, dtype=torch.float32), torch.tensor(yval, dtype=torch.long)),
    batch_size=batch_test, shuffle=False, pin_memory=pin, num_workers=0
)

perm_target = rng.permutation(len(X_target))
in_features = Xtr.shape[1]

# =====================
# PHASE 1: PRE-TRAIN
# =====================
print("\n" + "=" * 55)
print(f"  Phase 1: Pre-train (TinyCNN, BS{train_bss[0]}..BS{train_bss[-1]})")
print(f"  Val set: BS{val_bss} — target BS{target_bs} never seen")
print("=" * 55)

model_pre  = TinyCNN(in_features=in_features, num_classes=128).to(device)
opt_pre    = torch.optim.AdamW(model_pre.parameters(), lr=lr_pre, weight_decay=wd)
best_pre   = 0.0
best_state = None

for ep in range(1, epochs_pre + 1):
    t0 = time.time()
    model_pre.train()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_pre.zero_grad(set_to_none=True)
        loss = criterion(model_pre(xb), yb)
        loss.backward()
        opt_pre.step()

    accs = evaluate(model_pre, val_loader)
    if accs[1] > best_pre:
        best_pre = accs[1]
        best_state = copy.deepcopy(model_pre.state_dict())

    if ep % 5 == 0 or ep == 1:
        print(
            f"  [PRE] Ep{ep:02d} {time.time()-t0:4.1f}s | "
            f"val(BS11-13) top1 {accs[1]*100:5.2f}% | best {best_pre*100:5.2f}%"
        )

model_pre.load_state_dict(best_state)
torch.save(model_pre.state_dict(), "pretrained_bs3_10.pt")
pretrained_state = {k: v.cpu() for k, v in model_pre.state_dict().items()}
print(f"  Pre-train done. Best val top1: {best_pre*100:.2f}%")

# =====================
# PHASE 2: META-TRAIN (MTL STYLE)
# =====================
task_tensors = {
    bs: (
        torch.tensor(Xs_norm[i], dtype=torch.float32, device=device),
        torch.tensor(ys_list[i], dtype=torch.long, device=device)
    )
    for i, bs in enumerate(train_bss)
}

meta_val_tensors = {
    bs: (
        torch.tensor(Xval_list[i], dtype=torch.float32, device=device),
        torch.tensor(yval_list[i], dtype=torch.long, device=device)
    )
    for i, bs in enumerate(val_bss)
}

def sample_task_split(task_pool, n_support, n_query):
    bs_key = int(rng.choice(list(task_pool.keys())))
    X, y = task_pool[bs_key]
    idx = torch.from_numpy(
        rng.choice(len(X), n_support + n_query, replace=False)
    ).to(device)
    Xsup = X[idx[:n_support]]
    ysup = y[idx[:n_support]]
    Xqry = X[idx[n_support:]]
    yqry = y[idx[n_support:]]
    return bs_key, Xsup, ysup, Xqry, yqry

print("\n" + "=" * 55)
print("  Phase 2: Meta-train (MTL-style SS + classifier carry-over)")
print(f"  Train BS: {train_bss} — target BS{target_bs} excluded")
print("=" * 55)

model_meta = TinyCNNWithSS(in_features=in_features, num_classes=128).to(device)
load_tinycnn_weights_into_ss(model_meta, pretrained_state)
model_meta.reset_ss()
model_meta.freeze_backbone()

# Query-step optimizer updates BOTH SS and classifier, like paper Eq.4/Eq.5
meta_opt = torch.optim.AdamW(
    [
        {"params": model_meta.ss_params(),   "lr": meta_lr},
        {"params": model_meta.head_params(), "lr": meta_lr},
    ],
    weight_decay=wd
)
meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=meta_epochs)

best_meta_metric = float("inf")
best_meta_state  = copy.deepcopy(model_meta.state_dict())

for meta_ep in range(1, meta_epochs + 1):
    t0 = time.time()
    model_meta.train()

    support_losses = []
    query_losses   = []

    # Important:
    # theta and SS both persist across tasks inside the same training trajectory
    for _ in range(episodes_per_epoch):
        _, Xsup, ysup, Xqry, yqry = sample_task_split(task_tensors, n_shot_meta, n_query_meta)

        # Step A: support update on theta only, in-place, carried to next task
        for _ in range(inner_steps):
            logits_sup = model_meta(Xsup)
            loss_sup = F.cross_entropy(logits_sup, ysup)

            head_params = model_meta.head_params()
            grads = torch.autograd.grad(loss_sup, head_params, retain_graph=False, create_graph=False)

            with torch.no_grad():
                for p, g in zip(head_params, grads):
                    p -= inner_lr * g

            support_losses.append(loss_sup.item())

        # Step B: query update on SS + theta, also persistent
        meta_opt.zero_grad(set_to_none=True)
        logits_q = model_meta(Xqry)
        loss_q = F.cross_entropy(logits_q, yqry)
        loss_q.backward()
        meta_opt.step()

        query_losses.append(loss_q.item())

    meta_scheduler.step()

    train_sup_loss = float(np.mean(support_losses)) if support_losses else np.nan
    train_q_loss   = float(np.mean(query_losses)) if query_losses else np.nan

    if use_meta_val:
        model_meta.eval()
        val_losses = []

        for _ in range(meta_val_episodes):
            _, Xsup, ysup, Xqry, yqry = sample_task_split(meta_val_tensors, n_shot_meta, n_query_meta)

            # copy current state so validation does not alter training trajectory
            model_tmp = TinyCNNWithSS(in_features=in_features, num_classes=128).to(device)
            model_tmp.load_state_dict(copy.deepcopy(model_meta.state_dict()))
            model_tmp.freeze_backbone()
            model_tmp.train()

            # support update: theta only
            for _ in range(inner_steps):
                logits_sup = model_tmp(Xsup)
                loss_sup = F.cross_entropy(logits_sup, ysup)
                head_params = model_tmp.head_params()
                grads = torch.autograd.grad(loss_sup, head_params, retain_graph=False, create_graph=False)

                with torch.no_grad():
                    for p, g in zip(head_params, grads):
                        p -= inner_lr * g

            # query loss after support adaptation
            with torch.no_grad():
                logits_q = model_tmp(Xqry)
                loss_q = F.cross_entropy(logits_q, yqry)
            val_losses.append(loss_q.item())

        metric = float(np.mean(val_losses))
        metric_name = "meta-val query_loss"
    else:
        metric = train_q_loss
        metric_name = "train query_loss"

    if metric < best_meta_metric:
        best_meta_metric = metric
        best_meta_state = copy.deepcopy(model_meta.state_dict())

    if meta_ep % 50 == 0 or meta_ep == 1:
        print(
            f"  [META] Ep{meta_ep:04d} {time.time()-t0:4.1f}s | "
            f"support_loss {train_sup_loss:.4f} | "
            f"query_loss {train_q_loss:.4f} | "
            f"best {metric_name}: {best_meta_metric:.4f}"
        )

model_meta.load_state_dict(best_meta_state)
torch.save(model_meta.state_dict(), "meta_trained_mtl_ss_theta.pt")
print(f"  Meta-train done. Best metric: {best_meta_metric:.4f}")

# =====================
# PHASE 2b: META-TRAIN (MAML — ADDED)
# =====================
print("\n" + "=" * 55)
print("  Phase 2b: Meta-train (MAML from scratch)")
print(f"  Train BS: {train_bss} — target BS{target_bs} excluded")
print(f"  Val BS  : {val_bss}")
print("  Pretrain YOK")
print("=" * 55)

model_maml = TinyCNN(in_features=in_features, num_classes=128).to(device)
maml_opt = torch.optim.Adam(model_maml.parameters(), lr=maml_meta_lr)
maml_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(maml_opt, T_max=maml_epochs)

best_maml_metric = float("inf")
best_maml_state  = copy.deepcopy(model_maml.state_dict())

for maml_ep in range(1, maml_epochs + 1):
    t0 = time.time()
    model_maml.train()

    total_q_loss = torch.tensor(0.0, device=device)
    train_q_losses = []

    # true meta-batch → one outer step per epoch
    for _ in range(maml_episodes):
        _, Xsup, ysup, Xqry, yqry = sample_task_split(task_tensors, n_shot_meta, n_query_meta)

        fast_w = model_maml.get_fast_weights()

        # inner loop on full model
        for _ in range(maml_inner_steps):
            loss_s = F.cross_entropy(
                model_maml.forward_with_fast_weights(Xsup, fast_w), ysup
            )
            grads = torch.autograd.grad(
                loss_s,
                list(fast_w.values()),
                create_graph=True
            )
            fast_w = OrderedDict(
                (name, p - maml_inner_lr * g)
                for ((name, p), g) in zip(fast_w.items(), grads)
            )

        loss_q = F.cross_entropy(
            model_maml.forward_with_fast_weights(Xqry, fast_w), yqry
        )
        total_q_loss = total_q_loss + loss_q
        train_q_losses.append(loss_q.item())

    maml_opt.zero_grad(set_to_none=True)
    avg_q_loss = total_q_loss / maml_episodes
    avg_q_loss.backward()
    maml_opt.step()
    maml_scheduler.step()

    train_metric = float(np.mean(train_q_losses))

    # meta-val on BS11-13
    if use_meta_val:
        model_maml.eval()
        val_losses = []

        for _ in range(meta_val_episodes):
            _, Xsup, ysup, Xqry, yqry = sample_task_split(meta_val_tensors, n_shot_meta, n_query_meta)

            fast_w = OrderedDict(
                (name, p.detach().clone().requires_grad_(True))
                for name, p in model_maml.named_parameters()
            )

            for _ in range(maml_inner_steps):
                loss_s = F.cross_entropy(
                    model_maml.forward_with_fast_weights(Xsup, fast_w), ysup
                )
                grads = torch.autograd.grad(
                    loss_s,
                    list(fast_w.values()),
                    create_graph=False
                )
                fast_w = OrderedDict(
                    (name, p - maml_inner_lr * g)
                    for ((name, p), g) in zip(fast_w.items(), grads)
                )

            with torch.no_grad():
                loss_q = F.cross_entropy(
                    model_maml.forward_with_fast_weights(Xqry, fast_w), yqry
                )
            val_losses.append(loss_q.item())

        metric = float(np.mean(val_losses))
        metric_name = "meta-val query_loss"
    else:
        metric = train_metric
        metric_name = "train query_loss"

    if metric < best_maml_metric:
        best_maml_metric = metric
        best_maml_state = copy.deepcopy(model_maml.state_dict())

    if maml_ep % 50 == 0 or maml_ep == 1:
        print(
            f"  [MAML] Ep{maml_ep:04d} {time.time()-t0:4.1f}s | "
            f"train_q_loss {train_metric:.4f} | "
            f"best {metric_name}: {best_maml_metric:.4f}"
        )

model_maml.load_state_dict(best_maml_state)
torch.save(model_maml.state_dict(), "meta_trained_maml.pt")
print(f"  MAML meta-train done. Best metric: {best_maml_metric:.4f}")

# =====================
# PHASE 3: ADAPTATION SWEEP
# =====================
pool_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.long)
    ),
    batch_size=batch_test, shuffle=False, pin_memory=pin, num_workers=0
)
base_acc = evaluate(model_pre, pool_loader)
print(f"\n  BASE (no adapt) — BS{target_bs} pool top1: {base_acc[1]*100:.2f}%")

base_top1,   base_top2,   base_top3,   base_top5   = [], [], [], []
ftlast_top1, ftlast_top2, ftlast_top3, ftlast_top5 = [], [], [], []
ftall_top1,  ftall_top2,  ftall_top3,  ftall_top5  = [], [], [], []
ss_top1,     ss_top2,     ss_top3,     ss_top5     = [], [], [], []
maml_top1,   maml_top2,   maml_top3,   maml_top5   = [], [], [], []

meta_trained_mtl_state  = copy.deepcopy(model_meta.state_dict())
meta_trained_maml_state = copy.deepcopy(model_maml.state_dict())

print("\n" + "=" * 55)
print("  Phase 3: Adaptation sweep")
print(f"  Adapt: perm_target[:shots] | Test: perm_target[shots:]")
print("  No early stopping on target test")
print("=" * 55)

for shots in shots_list:
    adapt_idx = perm_target[:shots]
    test_idx  = perm_target[shots:]

    Xad, yad = X_target[adapt_idx], y_target[adapt_idx]
    Xte, yte = X_target[test_idx],  y_target[test_idx]

    adapt_loader = DataLoader(
        TensorDataset(torch.tensor(Xad, dtype=torch.float32), torch.tensor(yad, dtype=torch.long)),
        batch_size=min(batch_ft, shots), shuffle=True, pin_memory=pin, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long)),
        batch_size=batch_test, shuffle=False, pin_memory=pin, num_workers=0
    )

    # BASE
    acc0 = evaluate(model_pre, test_loader)
    base_top1.append(acc0[1]); base_top2.append(acc0[2])
    base_top3.append(acc0[3]); base_top5.append(acc0[5])

    # FT-LAST
    model_ftl = TinyCNN(in_features=in_features, num_classes=128).to(device)
    model_ftl.load_state_dict(pretrained_state)

    for p in model_ftl.parameters():
        p.requires_grad = False
    for p in model_ftl.head[-1].parameters():
        p.requires_grad = True

    opt_ftl = torch.optim.AdamW(model_ftl.head[-1].parameters(), lr=lr_ft_last, weight_decay=wd)

    for _ in range(ft_epochs(shots, "last")):
        model_ftl.train()
        for xb, yb in adapt_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_ftl.zero_grad(set_to_none=True)
            loss = criterion(model_ftl(xb), yb)
            loss.backward()
            opt_ftl.step()

    accs = evaluate(model_ftl, test_loader)
    ftlast_top1.append(accs[1]); ftlast_top2.append(accs[2])
    ftlast_top3.append(accs[3]); ftlast_top5.append(accs[5])

    # FT-ALL
    model_fta = TinyCNN(in_features=in_features, num_classes=128).to(device)
    model_fta.load_state_dict(pretrained_state)

    opt_fta = torch.optim.AdamW(model_fta.parameters(), lr=lr_ft_all, weight_decay=wd)

    for _ in range(ft_epochs(shots, "all")):
        model_fta.train()
        for xb, yb in adapt_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_fta.zero_grad(set_to_none=True)
            loss = criterion(model_fta(xb), yb)
            loss.backward()
            opt_fta.step()

    accs = evaluate(model_fta, test_loader)
    ftall_top1.append(accs[1]); ftall_top2.append(accs[2])
    ftall_top3.append(accs[3]); ftall_top5.append(accs[5])

    # MTL-SS + carried head checkpoint, then adapt SS + head on target support
    model_ss = TinyCNNWithSS(in_features=in_features, num_classes=128).to(device)
    model_ss.load_state_dict(meta_trained_mtl_state)
    model_ss.freeze_backbone()

    opt_ss = torch.optim.AdamW(
        model_ss.ss_params() + model_ss.head_params(),
        lr=lr_ss_adapt,
        weight_decay=wd
    )

    for _ in range(ss_adapt_epochs(shots)):
        model_ss.train()
        for xb, yb in adapt_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_ss.zero_grad(set_to_none=True)
            loss = criterion(model_ss(xb), yb)
            loss.backward()
            opt_ss.step()

    accs = evaluate(model_ss, test_loader)
    ss_top1.append(accs[1]); ss_top2.append(accs[2])
    ss_top3.append(accs[3]); ss_top5.append(accs[5])

    # ===== MAML ADDED =====
    model_maml_ad = TinyCNN(in_features=in_features, num_classes=128).to(device)
    model_maml_ad.load_state_dict(meta_trained_maml_state)

    opt_maml_ad = torch.optim.AdamW(model_maml_ad.parameters(), lr=lr_ft_all, weight_decay=wd)

    for _ in range(ft_epochs(shots, "all")):
        model_maml_ad.train()
        for xb, yb in adapt_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_maml_ad.zero_grad(set_to_none=True)
            loss = criterion(model_maml_ad(xb), yb)
            loss.backward()
            opt_maml_ad.step()

    accs = evaluate(model_maml_ad, test_loader)
    maml_top1.append(accs[1]); maml_top2.append(accs[2])
    maml_top3.append(accs[3]); maml_top5.append(accs[5])

    print(
        f"  [SHOTS {shots:5d}] "
        f"BASE {acc0[1]*100:5.2f}% | "
        f"FT-LAST {ftlast_top1[-1]*100:5.2f}% | "
        f"FT-ALL {ftall_top1[-1]*100:5.2f}% | "
        f"MTL-SS {ss_top1[-1]*100:5.2f}% | "
        f"MAML {maml_top1[-1]*100:5.2f}%"
    )

# =====================
# TABLE
# =====================
def fmt(v):
    return f"{v*100:8.2f}" if (v is not None and np.isfinite(v)) else "     nan"

for label, b, fl, fa, ss, ml in [
    ("Top-1", base_top1, ftlast_top1, ftall_top1, ss_top1, maml_top1),
    ("Top-2", base_top2, ftlast_top2, ftall_top2, ss_top2, maml_top2),
    ("Top-3", base_top3, ftlast_top3, ftall_top3, ss_top3, maml_top3),
    ("Top-5", base_top5, ftlast_top5, ftall_top5, ss_top5, maml_top5),
]:
    print(f"\n{label}")
    print(f"{'shots':>6} | {'BASE':>8} | {'FT-LAST':>8} | {'FT-ALL':>8} | {'MTL-SS':>8} | {'MAML':>8}")
    print("-" * 67)
    for i, s in enumerate(shots_list):
        print(f"{s:6d} | {fmt(b[i])} | {fmt(fl[i])} | {fmt(fa[i])} | {fmt(ss[i])} | {fmt(ml[i])}")

# =====================
# PLOT
# =====================
fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
topk_results = [
    (1, base_top1, ftlast_top1, ftall_top1, ss_top1, maml_top1),
    (2, base_top2, ftlast_top2, ftall_top2, ss_top2, maml_top2),
    (3, base_top3, ftlast_top3, ftall_top3, ss_top3, maml_top3),
    (5, base_top5, ftlast_top5, ftall_top5, ss_top5, maml_top5),
]

for ax, (k, base, ftlast, ftall, ss, ml) in zip(axes, topk_results):
    ax.axhline(base_acc[k] * 100, color='k', ls='--', lw=1.5,
               label=f'BASE ({base_acc[k]*100:.1f}%)')
    ax.plot(shots_list, [v * 100 for v in ftlast], marker='^', label='FT-LAST')
    ax.plot(shots_list, [v * 100 for v in ftall],  marker='s', label='FT-ALL')
    ax.plot(shots_list, [v * 100 for v in ss],     marker='D', lw=2, label='MTL-SS + carried head')
    ax.plot(shots_list, [v * 100 for v in ml],     marker='o', lw=2, ls='-.', label='MAML (no pretrain)')
    ax.set_xscale('log')
    ax.set_xlabel('Adaptation samples [log]')
    ax.set_ylabel(f'Top-{k} Accuracy (%)')
    ax.set_title(f'Top-{k} — BS{target_bs} Test')
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=8)

plt.suptitle(
    f'BS{target_bs} Few-Shot Adaptation: 5-Method Comparison\n'
    f'(Pre-train: BS3–BS10, Val: BS11–13, Meta-train: MTL-style SS + theta carry-over, MAML from scratch)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(f"results_bs{target_bs}_5methods.png", dpi=150)
plt.show()
print(f"\nPlot saved: results_bs{target_bs}_5methods.png")
