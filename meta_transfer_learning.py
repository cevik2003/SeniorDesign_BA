"""
Meta-Transfer Learning (MTL) for Beam Selection
================================================
Phase 1  — Pre-train  : BeamNet trained on BS3–BS13 (128-class, large-scale data).
           Feature extractor Θ + Classifier θ jointly optimised (Eq.1, 2).
           Θ is then frozen. θ is kept (128 classes fixed → never discarded).

Phase 2  — Meta-train : Each outdoor BS station (BS3–BS13) = one task T_i.
           Learnable SS parameters Φ_S initialised: scale α=1, shift β=0.
           For every task T_i with support T^(tr) and query T^(te):
             Eq.(3)  θ'  ← θ  − β  ∇_θ  L_{T^(tr)}([Θ; θ ], Φ_S)   # adapt classifier
             Eq.(4)  Φ_S ← Φ_S − γ₁ ∇_{Φ_S} L_{T^(te)}([Θ; θ'], Φ_S)  # update SS
             Eq.(5)  θ   ← θ   − γ₂ ∇_θ   L_{T^(te)}([Θ; θ'], Φ_S)   # update classifier
           θ and Φ_S carry to the next task.

Phase 3  — Meta-test  : Adapt classifier on BS14/BS15 support, evaluate on query.

Hyperparameters match the original MTL repo (pytorch/main.py):
  meta_lr1 = 0.0001   (γ₁, SS params)
  meta_lr2 = 0.001    (γ₂, classifier θ)
  base_lr  = 0.01     (β,  inner-loop)
  update_step = 3     (inner-loop gradient steps)
  step_size   = 10    (LR scheduler epoch step)
  gamma       = 0.5   (LR decay factor)
"""

import os
import math
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "."
OUTDOOR_BSS = list(range(3, 14))   # BS3–BS13 : pretrain + meta-train tasks
INDOOR_BSS  = [14, 15]             # BS14–BS15 : meta-test
TOP_K       = 50_000
NUM_CLASSES = 128
INPUT_H, INPUT_W = 4, 8            # reshape 32-dim signal → (1, 4, 8)
KEY_INPUT   = "X"
KEY_LABEL   = "best128"

# Phase 1 – Pre-training (matches baseline.py)
PRETRAIN_BATCH  = 256
PRETRAIN_EPOCHS = 100
PRETRAIN_LR     = 1e-3
PRETRAIN_CKPT   = "pretrained_beamnet.pth"

# Phase 2 – Meta-training (matches repo pytorch/main.py defaults)
META_LR1        = 0.0001   # γ₁ : learning rate for SS params  (repo: meta_lr1)
META_LR2        = 0.001    # γ₂ : learning rate for classifier θ (repo: meta_lr2)
BASE_LR         = 0.01     # β  : inner-loop LR for Eq.(3)       (repo: base_lr)
UPDATE_STEP     = 3        # inner-loop gradient steps            (repo: update_step)
META_EPOCHS     = 100      # outer meta-training epochs
N_SHOTS_TRAIN   = 10       # support samples per class per task
QUERY_SAMPLES   = 500      # max query samples per task (capped for memory)
STEP_SIZE       = 10       # LR scheduler step (repo: step_size)
LR_GAMMA        = 0.5      # LR decay factor   (repo: gamma)
META_CKPT       = "meta_transfer_beamnet.pth"

# Phase 3 – Meta-test
SHOT_COUNTS     = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
FINETUNE_LR     = 1e-3
FINETUNE_EPOCHS = 50
SEED            = 42

# ── Execution phase: change to run different phases ───────────────────────────
# Options: "pretrain" | "meta_train" | "meta_test" | "all"
PHASE = "all"


# ── Dataset ───────────────────────────────────────────────────────────────────

class BeamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_raw(bs_idx: int, is_outdoor: bool):
    """Load raw X (N,32) and labels (N,) for a given BS."""
    path = os.path.join(DATA_DIR, f"beam_dataset_bs{bs_idx}.mat")
    data = sio.loadmat(path)
    X = data[KEY_INPUT].astype(np.float32)
    y = data[KEY_LABEL].astype(np.int64).flatten()
    if X.shape[1] != 32:
        X = X.T
    if is_outdoor:
        top_idx = np.argsort(X.max(axis=1))[-TOP_K:]
        X, y = X[top_idx], y[top_idx]
    assert y.min() >= 0 and y.max() < NUM_CLASSES, \
        f"BS{bs_idx}: label out of range [{y.min()}, {y.max()}]"
    return X, y


def make_dataset(X: np.ndarray, y: np.ndarray,
                 mu: np.ndarray, std: np.ndarray) -> BeamDataset:
    """Normalise with pre-computed train stats, reshape to (N,1,4,8)."""
    X = (X - mu) / std
    X = X.reshape(-1, 1, INPUT_H, INPUT_W)
    return BeamDataset(X.astype(np.float32), y)


# ── Conv2dMtl ─────────────────────────────────────────────────────────────────
# Based on pytorch/models/conv2d_mtl.py (Yaoyao Liu, 2019)
# SS operation (Eq.6): SS(X; W, b; Φ_S) = (W ⊙ α)X + (b + β_SS)

class Conv2dMtl(nn.Module):
    """
    Conv2d with meta-transfer Scaling & Shifting.
      - Base weight W and bias b → frozen after loading pre-trained values.
      - mtl_weight (α)  : learnable scale,  initialised to 1.  Shape (out, in, 1, 1).
      - mtl_bias  (β_SS): learnable shift, initialised to 0.  Shape (out_channels,).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=3, stride=1, padding=0):
        super().__init__()
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride  = stride
        self.padding = padding

        # Frozen base weights (Θ component for this layer)
        self.weight = Parameter(torch.empty(out_channels, in_channels, *ks))
        self.weight.requires_grad = False
        self.bias = Parameter(torch.zeros(out_channels))
        self.bias.requires_grad = False

        # Learnable SS parameters (Φ_S component for this layer)
        self.mtl_weight = Parameter(torch.ones(out_channels, in_channels, 1, 1))
        self.mtl_bias   = Parameter(torch.zeros(out_channels))

        # Initialise base weights (will be overwritten by load_pretrained)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * ks[0] * ks[1]
        bound  = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W_eff = W ⊙ α  (element-wise, α broadcast over spatial dims)
        w_eff = self.weight * self.mtl_weight.expand_as(self.weight)
        b_eff = self.bias + self.mtl_bias
        return F.conv2d(x, w_eff, b_eff, self.stride, self.padding)

    def copy_from_conv2d(self, src: nn.Conv2d):
        """Copy pre-trained base weights from a standard nn.Conv2d."""
        with torch.no_grad():
            self.weight.copy_(src.weight)
            if src.bias is not None:
                self.bias.copy_(src.bias)


# ── BeamNet (standard, used for Phase 1 pre-training) ────────────────────────

class BeamNet(nn.Module):
    """
    Standard BeamNet for Phase 1 (identical to baseline.py).
    Conv1: 1→32  (3×3, pad 1) + ReLU
    Conv2: 32→64 (3×3, pad 1) + ReLU
    Conv3: 64→128(1×1, pad 0) + ReLU
    FC   : 128*4*8 → 128 classes
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,   32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32,  64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 1, 1, 0), nn.ReLU(),
        )
        self.classifier = nn.Linear(128 * INPUT_H * INPUT_W, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


# ── BeamNetMtl (Phase 2 meta-transfer model) ──────────────────────────────────

class BeamNetMtl(nn.Module):
    """
    BeamNet with MTL Scaling & Shifting on the feature extractor.
    Θ (base conv weights) → frozen.
    Φ_S (mtl_weight, mtl_bias) → trainable during meta-training.
    θ (classifier FC) → kept and updated across tasks (never discarded).
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 base_lr: float = BASE_LR,
                 update_step: int = UPDATE_STEP):
        super().__init__()
        self.base_lr     = base_lr
        self.update_step = update_step

        # Feature extractor with SS layers
        self.conv1 = Conv2dMtl(1,   32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2dMtl(32,  64, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2dMtl(64, 128, kernel_size=1, stride=1, padding=0)
        self.relu  = nn.ReLU(inplace=True)

        # Classifier θ: 128-class, never discarded
        self.classifier = nn.Linear(128 * INPUT_H * INPUT_W, num_classes)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through frozen Θ + learnable Φ_S → flat embedding."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x.flatten(start_dim=1)          # (N, 128*4*8)

    @staticmethod
    def _classify(emb: torch.Tensor,
                  fc_w: torch.Tensor,
                  fc_b: torch.Tensor) -> torch.Tensor:
        return F.linear(emb, fc_w, fc_b)

    # ── Public forwards ───────────────────────────────────────────────────────

    def pretrain_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — all parameters trainable (Phase 1 only)."""
        return self._classify(self._encode(x),
                              self.classifier.weight,
                              self.classifier.bias)

    def meta_forward(self, X_support: torch.Tensor,
                     y_support: torch.Tensor,
                     X_query: torch.Tensor) -> torch.Tensor:
        """
        Meta-transfer forward for one task.

        Inner loop  Eq.(3): θ' ← θ − β ∇_θ L_{T^(tr)}([Θ; θ], Φ_S)
                            (repeated update_step times)
        Returns query logits computed with (Θ, θ', Φ_S).
        When outer loss.backward() is called, gradients flow to:
          • Φ_S via query path                          → satisfies Eq.(4)
          • θ   via fast_weights ← θ (higher-order)     → satisfies Eq.(5)
        """
        emb_s = self._encode(X_support)
        emb_q = self._encode(X_query)

        fc_w = self.classifier.weight   # θ weight
        fc_b = self.classifier.bias     # θ bias

        # ── Inner loop step 1 (Eq.3 first iteration) ─────────────────────────
        logits_s = self._classify(emb_s, fc_w, fc_b)
        loss_s   = F.cross_entropy(logits_s, y_support)
        # create_graph=True: allows outer grad to flow back through inner grad
        grad = torch.autograd.grad(loss_s, [fc_w, fc_b], create_graph=True)
        fast_w = fc_w - self.base_lr * grad[0]
        fast_b = fc_b - self.base_lr * grad[1]

        # ── Additional inner loop steps ───────────────────────────────────────
        for _ in range(1, self.update_step):
            logits_s = self._classify(emb_s, fast_w, fast_b)
            loss_s   = F.cross_entropy(logits_s, y_support)
            grad     = torch.autograd.grad(loss_s, [fast_w, fast_b],
                                           create_graph=True)
            fast_w = fast_w - self.base_lr * grad[0]
            fast_b = fast_b - self.base_lr * grad[1]

        # ── Query logits with adapted θ' (fast_w, fast_b) ────────────────────
        logits_q = self._classify(emb_q, fast_w, fast_b)
        return logits_q

    # ── Parameter groups ──────────────────────────────────────────────────────

    def ss_parameters(self):
        """Φ_S: learnable scaling & shifting params (meta-learner)."""
        return [p for n, p in self.named_parameters()
                if 'mtl_weight' in n or 'mtl_bias' in n]

    def classifier_parameters(self):
        """θ: classifier FC parameters (base-learner, meta-level)."""
        return list(self.classifier.parameters())

    def freeze_base_weights(self):
        """Freeze all base conv weights (Θ). SS params remain trainable."""
        for layer in [self.conv1, self.conv2, self.conv3]:
            layer.weight.requires_grad = False
            layer.bias.requires_grad   = False

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_pretrained(self, ckpt_path: str):
        """
        Load Phase 1 BeamNet checkpoint into BeamNetMtl.
        Maps standard Conv2d weights → Conv2dMtl frozen base weights.
        Returns (mu, std) for data normalisation.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model', ckpt)

        # features.0 → conv1, features.2 → conv2, features.4 → conv3
        mapping = [
            ('features.0.weight', self.conv1.weight),
            ('features.0.bias',   self.conv1.bias),
            ('features.2.weight', self.conv2.weight),
            ('features.2.bias',   self.conv2.bias),
            ('features.4.weight', self.conv3.weight),
            ('features.4.bias',   self.conv3.bias),
            ('classifier.weight', self.classifier.weight),
            ('classifier.bias',   self.classifier.bias),
        ]
        with torch.no_grad():
            for src_key, dst in mapping:
                if src_key in sd:
                    dst.copy_(sd[src_key])
                else:
                    print(f"  [warn] key not found: {src_key}")

        mu  = ckpt.get('mu',  None)
        std = ckpt.get('std', None)
        return mu, std


# ── Utility functions ─────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


@torch.no_grad()
def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    topk = logits.topk(k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()


def sample_task(X_np: np.ndarray, y_np: np.ndarray,
                n_shots: int, max_query: int, rng: np.random.Generator):
    """
    Sample a few-shot task from a BS dataset.
    Returns (support_X, support_y, query_X, query_y) as numpy arrays.
    """
    support_idx, query_idx = [], []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(y_np == cls)[0]
        if len(cls_idx) < n_shots + 1:
            continue  # need at least 1 query sample
        supp = rng.choice(cls_idx, n_shots, replace=False)
        rest = np.setdiff1d(cls_idx, supp)
        support_idx.append(supp)
        query_idx.append(rest)

    support_idx = np.concatenate(support_idx)
    query_idx   = np.concatenate(query_idx)

    if len(query_idx) > max_query:
        query_idx = rng.choice(query_idx, max_query, replace=False)

    return (X_np[support_idx], y_np[support_idx],
            X_np[query_idx],   y_np[query_idx])


# ── Phase 1: Pre-training ─────────────────────────────────────────────────────

def pretrain(device: torch.device):
    """
    Train a standard BeamNet on BS3–BS13 (all data, 128 classes).
    Implements Eq.(1) and Eq.(2) from the paper.
    Saves checkpoint to PRETRAIN_CKPT.
    Returns the trained BeamNet and (mu, std).
    """
    print("=" * 60)
    print("PHASE 1 — Pre-training on BS3–BS13 (128-class)")
    print("=" * 60)

    # Load and concatenate training data
    print("Loading BS3–BS13 …")
    train_Xs, train_ys = [], []
    for bs in OUTDOOR_BSS:
        X, y = load_raw(bs, is_outdoor=True)
        train_Xs.append(X)
        train_ys.append(y)
        print(f"  BS{bs}: {len(y):,} samples")

    X_train = np.concatenate(train_Xs)
    y_train = np.concatenate(train_ys)
    print(f"  Total : {len(y_train):,}\n")

    # Compute normalisation stats from training data only
    mu  = X_train.mean(axis=0, keepdims=True)           # (1, 32)
    std = X_train.std(axis=0,  keepdims=True) + 1e-8    # (1, 32)

    train_ds  = make_dataset(X_train, y_train, mu, std)
    pin_mem   = device.type == "cuda"
    loader    = DataLoader(train_ds, batch_size=PRETRAIN_BATCH,
                           shuffle=True, num_workers=0, pin_memory=pin_mem)

    model     = BeamNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    print(f"{'Epoch':>6}  {'Loss':>10}  {'Acc':>8}")
    print("─" * 30)

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)
        if epoch % 10 == 0:
            print(f"{epoch:6d}  {total_loss/n:10.4f}  {correct/n*100:7.2f}%")

    # Save checkpoint (model weights + normalisation stats)
    torch.save({"model": model.state_dict(), "mu": mu, "std": std},
               PRETRAIN_CKPT)
    print(f"\nCheckpoint saved → {PRETRAIN_CKPT}\n")
    return model, mu, std


# ── Phase 2: Meta-Transfer Learning ──────────────────────────────────────────

def meta_train(device: torch.device):
    """
    Phase 2: Meta-transfer learning.
    Loads pre-trained weights, freezes Θ, learns Φ_S and θ across tasks.
    Each outdoor BS station is one task.
    Update order: Eq.(3) → Eq.(4) → Eq.(5) per task.
    """
    print("=" * 60)
    print("PHASE 2 — Meta-Transfer Learning")
    print("=" * 60)

    if not os.path.exists(PRETRAIN_CKPT):
        raise FileNotFoundError(
            f"Pre-trained checkpoint not found: {PRETRAIN_CKPT}\n"
            "Run Phase 1 first (PHASE = 'pretrain').")

    # ── Load BeamNetMtl with pre-trained weights ───────────────────────────
    model = BeamNetMtl(base_lr=BASE_LR, update_step=UPDATE_STEP).to(device)
    mu, std = model.load_pretrained(PRETRAIN_CKPT)
    print(f"Loaded pre-trained weights from {PRETRAIN_CKPT}")

    # Freeze base conv weights (Θ)
    model.freeze_base_weights()
    print("Feature extractor Θ frozen. SS params Φ_S + classifier θ trainable.\n")

    # ── Prepare task data (each BS = one task) ────────────────────────────
    print("Preparing task datasets (BS3–BS13) …")
    task_data = []
    for bs in OUTDOOR_BSS:
        X, y = load_raw(bs, is_outdoor=True)
        X_norm = ((X - mu) / std).reshape(-1, 1, INPUT_H, INPUT_W).astype(np.float32)
        task_data.append((X_norm, y))
        print(f"  BS{bs}: {len(y):,} samples")
    print()

    # ── Optimizer: separate LRs for Φ_S (meta_lr1) and θ (meta_lr2) ──────
    # Matches repo's MetaTrainer optimizer setup (pytorch/trainer/meta.py)
    optimizer = torch.optim.Adam([
        {'params': model.ss_parameters(),       'lr': META_LR1},
        {'params': model.classifier_parameters(), 'lr': META_LR2},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=LR_GAMMA)

    rng = np.random.default_rng(SEED)

    best_loss = float('inf')
    print(f"{'Epoch':>6}  {'Avg Loss':>10}  {'Avg Acc':>9}  {'LR-SS':>10}  {'LR-θ':>10}")
    print("─" * 55)

    for epoch in range(1, META_EPOCHS + 1):
        scheduler.step()
        model.train()

        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_tasks    = len(task_data)

        for X_bs, y_bs in task_data:
            # ── Sample support and query for this task ─────────────────────
            X_s, y_s, X_q, y_q = sample_task(
                X_bs, y_bs, N_SHOTS_TRAIN, QUERY_SAMPLES, rng)

            X_s = torch.from_numpy(X_s).to(device)
            y_s = torch.from_numpy(y_s).to(device)
            X_q = torch.from_numpy(X_q).to(device)
            y_q = torch.from_numpy(y_q).to(device)

            # ── Meta-forward ───────────────────────────────────────────────
            # Implements Eq.(3) internally (inner loop)
            # Returns query logits computed with adapted θ' and current Φ_S
            logits_q = model.meta_forward(X_s, y_s, X_q)

            # ── Outer loss L_{T^(te)} ──────────────────────────────────────
            loss_q = F.cross_entropy(logits_q, y_q)

            # ── Eq.(4): update Φ_S    Φ_S ← Φ_S − γ₁ ∇_{Φ_S} L_{T^(te)} ─
            # ── Eq.(5): update θ      θ   ← θ   − γ₂ ∇_θ    L_{T^(te)} ──
            # Both updates are handled by optimizer.step() with separate LRs.
            # loss_q.backward() computes:
            #   ∂L_q/∂Φ_S (through emb_q path + higher-order emb_s path)
            #   ∂L_q/∂θ   (through fast_weights back to θ)
            optimizer.zero_grad()
            loss_q.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss_q.item()
                epoch_acc  += accuracy(logits_q, y_q)

        avg_loss = epoch_loss / n_tasks
        avg_acc  = epoch_acc  / n_tasks

        if epoch % 10 == 0:
            lr_ss  = optimizer.param_groups[0]['lr']
            lr_clf = optimizer.param_groups[1]['lr']
            print(f"{epoch:6d}  {avg_loss:10.4f}  {avg_acc*100:8.2f}%  "
                  f"{lr_ss:10.6f}  {lr_clf:10.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "mu":    mu,
                "std":   std,
                "epoch": epoch,
            }, META_CKPT)

    print(f"\nBest checkpoint saved → {META_CKPT} (loss {best_loss:.4f})\n")
    return model, mu, std


# ── Phase 3: Meta-test evaluation ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(model: BeamNetMtl, loader: DataLoader,
                    device: torch.device):
    """Evaluate model accuracy (top-1 and top-3) on a DataLoader."""
    model.eval()
    correct1, correct3, n = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model.pretrain_forward(X)
        correct1 += (logits.argmax(1) == y).sum().item()
        top3      = logits.topk(3, dim=1).indices
        correct3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
        n        += len(y)
    return correct1 / n, correct3 / n


def meta_test(device: torch.device):
    """
    Phase 3: Meta-test on BS14 and BS15.
    For each shot count, sample a support set, fine-tune classifier θ
    (Φ_S and Θ remain frozen), evaluate on query set.
    """
    print("=" * 60)
    print("PHASE 3 — Meta-Test on BS14 & BS15")
    print("=" * 60)

    if not os.path.exists(META_CKPT):
        raise FileNotFoundError(
            f"Meta-trained checkpoint not found: {META_CKPT}\n"
            "Run Phase 2 first (PHASE = 'meta_train').")

    # ── Load meta-trained model ────────────────────────────────────────────
    model = BeamNetMtl(base_lr=BASE_LR, update_step=UPDATE_STEP).to(device)
    ckpt  = torch.load(META_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    mu  = ckpt['mu']
    std = ckpt['std']
    print(f"Loaded meta-trained model from {META_CKPT} (epoch {ckpt['epoch']})\n")

    # ── Load merged indoor data (BS14+BS15) ───────────────────────────────
    print("Loading test data (BS14 + BS15) …")
    indoor_Xs, indoor_ys = [], []
    for bs in INDOOR_BSS:
        X, y = load_raw(bs, is_outdoor=False)
        indoor_Xs.append(X)
        indoor_ys.append(y)
        print(f"  BS{bs}: {len(y):,} samples")
    X_indoor = np.concatenate(indoor_Xs)
    y_indoor = np.concatenate(indoor_ys)
    print(f"  Merged: {len(y_indoor):,}\n")

    full_ds  = make_dataset(X_indoor, y_indoor, mu, std)
    pin_mem  = device.type == "cuda"
    full_loader = DataLoader(full_ds, batch_size=256,
                             num_workers=0, pin_memory=pin_mem)
    criterion   = nn.CrossEntropyLoss()

    # ── Zero-shot baseline (no fine-tuning) ───────────────────────────────
    base_acc1, base_acc3 = evaluate_loader(model, full_loader, device)
    print(f"0-shot (no fine-tune)  top-1: {base_acc1*100:.2f}%  "
          f"top-3: {base_acc3*100:.2f}%\n")

    rng     = np.random.default_rng(SEED)
    N_total = len(full_ds)
    all_idx = np.arange(N_total)

    print(f"{'Shots':>6}  {'Support':>8}  {'Query':>8}  {'Top-1 (%)':>10}  {'Top-3 (%)':>10}")
    print("─" * 52)

    results = {}
    for n_shots in SHOT_COUNTS:
        if n_shots >= N_total:
            print(f"{n_shots:6d}  -- not enough samples --")
            continue

        # Sample support / query split
        support_idx = rng.choice(all_idx, size=n_shots, replace=False)
        query_idx   = np.setdiff1d(all_idx, support_idx)

        from torch.utils.data import Subset
        support_ds = Subset(full_ds, support_idx.tolist())
        query_ds   = Subset(full_ds, query_idx.tolist())
        support_loader = DataLoader(support_ds, batch_size=min(n_shots, 64),
                                    shuffle=True, num_workers=0)
        query_loader   = DataLoader(query_ds, batch_size=256,
                                    num_workers=0, pin_memory=pin_mem)

        # Fine-tune only classifier θ; Θ and Φ_S remain frozen
        import copy
        ft_model = copy.deepcopy(model)
        # Freeze everything except classifier
        for p in ft_model.ss_parameters():
            p.requires_grad = False
        ft_optimizer = torch.optim.Adam(
            ft_model.classifier_parameters(), lr=FINETUNE_LR)

        ft_model.train()
        for _ in range(FINETUNE_EPOCHS):
            for X, y in support_loader:
                X, y = X.to(device), y.to(device)
                ft_optimizer.zero_grad()
                logits = ft_model.pretrain_forward(X)
                F.cross_entropy(logits, y).backward()
                ft_optimizer.step()

        acc1, acc3 = evaluate_loader(ft_model, query_loader, device)
        results[n_shots] = (acc1, acc3)
        print(f"{n_shots:6d}  {n_shots:8d}  {len(query_idx):8,}  "
              f"{acc1*100:10.2f}%  {acc3*100:10.2f}%")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  {'Shots':>6}  {'Top-1 (%)':>10}  {'Top-3 (%)':>10}")
    print(f"  {'0 (base)':>6}  {base_acc1*100:9.2f}%  {base_acc3*100:9.2f}%")
    for n_shots, (acc1, acc3) in results.items():
        print(f"  {n_shots:6d}  {acc1*100:9.2f}%  {acc3*100:9.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    if PHASE in ("pretrain", "all"):
        pretrain(device)

    if PHASE in ("meta_train", "all"):
        meta_train(device)

    if PHASE in ("meta_test", "all"):
        meta_test(device)


if __name__ == "__main__":
    main()
