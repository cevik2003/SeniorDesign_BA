"""
Transfer Learning: Beam Selection via CNN (Last-Layer Fine-Tuning)
Google Colab single-script — no external checkpoint needed.

Pretrain : BS3–BS13 outdoor (top-50k per BS), same as baseline
Fine-tune: freeze conv features, update only final FC layer
           using N samples from merged BS14+BS15 indoor test data
Evaluate : remaining merged BS14+BS15 samples (support set excluded)

Shot counts: 10, 20, 30, ..., 100, 200, 500

── Colab usage ──────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/<your-folder>"   # ← change this
─────────────────────────────────────────────────────────────────────────────
"""

import os
import copy
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "."                  # ← Colab'da Drive path ile değiştir
OUTDOOR_BSS  = list(range(3, 14))  # BS3–BS13  (pretrain)
INDOOR_BSS   = [14, 15]            # BS14–BS15 (fine-tune + test)
TOP_K        = 50_000
BATCH_SIZE   = 256
PRETRAIN_EPOCHS = 100
PRETRAIN_LR     = 1e-3
FINETUNE_EPOCHS = 50
FINETUNE_LR     = 1e-3
NUM_CLASSES  = 128
IN_FEATURES  = 32

KEY_INPUT = "X"
KEY_LABEL = "best128"

SHOT_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
SEED        = 42


# ── Dataset ───────────────────────────────────────────────────────────────────
class BeamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_raw(bs_idx: int, is_outdoor: bool):
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
    X = (X - mu) / std
    return BeamDataset(X, y)


# ── Model ─────────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, in_features=32, num_classes=NUM_CLASSES):
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


# ── Train / Eval loops ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


# ── Pretrain ──────────────────────────────────────────────────────────────────
def pretrain(device):
    print("=" * 50)
    print("PRETRAINING on BS3–BS13 …")
    print("=" * 50)

    train_Xs, train_ys = [], []
    for bs in OUTDOOR_BSS:
        X, y = load_raw(bs, is_outdoor=True)
        train_Xs.append(X)
        train_ys.append(y)
    X_train = np.concatenate(train_Xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)

    mu  = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0,  keepdims=True) + 1e-8

    train_ds     = make_dataset(X_train, y_train, mu, std)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=pin)
    print(f"  Train samples : {len(train_ds):,}\n")

    model     = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}")
    print("─" * 32)
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % 10 == 0:
            print(f"{epoch:6d}  {tr_loss:10.4f}  {tr_acc*100:9.2f}%")

    print()
    return model, mu, std


# ── Transfer Learning ─────────────────────────────────────────────────────────
def run_transfer(device):
    # ── Pretrain ──────────────────────────────────────────────────────────────
    base_model, mu, std = pretrain(device)

    criterion = nn.CrossEntropyLoss()

    # ── Baseline accuracy (no fine-tuning) on merged indoor data ─────────────
    print("Loading merged test data (BS14 + BS15) …")
    indoor_Xs, indoor_ys = [], []
    for bs in INDOOR_BSS:
        X, y = load_raw(bs, is_outdoor=False)
        indoor_Xs.append(X)
        indoor_ys.append(y)
        print(f"  BS{bs} samples : {len(y):,}")
    X_indoor = np.concatenate(indoor_Xs, axis=0)
    y_indoor = np.concatenate(indoor_ys, axis=0)
    print(f"  Merged total  : {len(y_indoor):,}\n")

    full_ds = make_dataset(X_indoor, y_indoor, mu, std)
    pin = device.type == "cuda"
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE,
                             num_workers=0, pin_memory=pin)

    _, base_acc = evaluate(base_model, full_loader, criterion, device)
    print(f"Baseline (0-shot, no fine-tune) accuracy : {base_acc*100:.2f}%\n")

    # ── Last-layer fine-tuning sweep ──────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    N_total = len(full_ds)
    all_indices = np.arange(N_total)

    print(f"{'Shots':>6}  {'Support':>8}  {'Query':>8}  {'Acc (%)':>10}")
    print("─" * 40)

    results = {}
    for n_shots in SHOT_COUNTS:
        if n_shots >= N_total:
            print(f"{n_shots:6d}  -- not enough samples --")
            continue

        # Sample support set indices
        support_idx = rng.choice(all_indices, size=n_shots, replace=False)
        query_idx   = np.setdiff1d(all_indices, support_idx)

        support_ds = Subset(full_ds, support_idx.tolist())
        query_ds   = Subset(full_ds, query_idx.tolist())

        support_loader = DataLoader(support_ds, batch_size=min(n_shots, 64),
                                    shuffle=True, num_workers=0)
        query_loader   = DataLoader(query_ds, batch_size=BATCH_SIZE,
                                    num_workers=0, pin_memory=pin)

        # Deep-copy base model, freeze cnn+first head layer, only train last linear
        model = copy.deepcopy(base_model)
        for param in model.cnn.parameters():
            param.requires_grad = False
        for param in model.head[0].parameters():
            param.requires_grad = False
        for param in model.head[2].parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR
        )

        for _ in range(FINETUNE_EPOCHS):
            train_epoch(model, support_loader, optimizer, criterion, device)

        _, acc = evaluate(model, query_loader, criterion, device)
        results[n_shots] = acc

        print(f"{n_shots:6d}  {n_shots:8d}  {len(query_idx):8,}  {acc*100:10.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────────")
    print(f"  {'Shots':>6}  {'Accuracy':>10}")
    print(f"  {'0 (base)':>6}  {base_acc*100:9.2f}%")
    for n_shots, acc in results.items():
        print(f"  {n_shots:6d}  {acc*100:9.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")
    run_transfer(device)


if __name__ == "__main__":
    main()
