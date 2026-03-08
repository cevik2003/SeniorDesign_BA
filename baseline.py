"""
Baseline: Beam Selection via CNN
Train : BS3–BS13 (outdoor, top-50k users per BS)
Test  : BS14–BS15 (indoor, all users)

Input  : R^32  (received powers from 32 wide beams)
Label  : best128 — integer beam index (1-indexed in MATLAB → 0-indexed here)
"""

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "."
OUTDOOR_BSS  = list(range(3, 14))   # BS3–BS13  (train)
INDOOR_BSS   = [14, 15]             # BS14–BS15 (test)
TOP_K        = 50_000
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
NUM_CLASSES  = 128
INPUT_H, INPUT_W = 4, 8            # reshape 32 → (1, 4, 8)

KEY_INPUT = "X"
KEY_LABEL = "best128"


# ── Dataset ───────────────────────────────────────────────────────────────────
class BeamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, 1, 4, 8)
        self.y = torch.from_numpy(y)   # (N,) int64

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_raw(bs_idx: int, is_outdoor: bool):
    """Load raw X (N,32) and labels (N,) without normalization."""
    path = os.path.join(DATA_DIR, f"beam_dataset_bs{bs_idx}.mat")
    data = sio.loadmat(path)

    X = data[KEY_INPUT].astype(np.float32)          # (N, 32)
    y = data[KEY_LABEL].astype(np.int64).flatten()  # (N,)

    if X.shape[1] != 32:
        X = X.T

    if is_outdoor:
        top_idx = np.argsort(X.max(axis=1))[-TOP_K:]
        X, y    = X[top_idx], y[top_idx]

    labels = y - 1  # 1-indexed → 0-indexed
    assert labels.min() >= 0 and labels.max() < NUM_CLASSES, \
        f"BS{bs_idx}: label out of range [{labels.min()}, {labels.max()}]"

    return X, labels


def make_dataset(X: np.ndarray, labels: np.ndarray, mu: np.ndarray, std: np.ndarray) -> BeamDataset:
    """Normalize with pre-computed train stats and return dataset."""
    X = (X - mu) / std
    X = X.reshape(-1, 1, INPUT_H, INPUT_W)
    return BeamDataset(X, labels)


# ── Model ─────────────────────────────────────────────────────────────────────
class BeamNet(nn.Module):
    """
    Conv1: 1  → 32  filters, 3×3, stride 1, pad 1  + ReLU
    Conv2: 32 → 64  filters, 3×3, stride 1, pad 1  + ReLU
    Conv3: 64 → 128 filters, 1×1, stride 1, pad 0  + ReLU
    FC   : 128*4*8 → 128 classes
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,   32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32,  64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0), nn.ReLU(),
        )
        self.classifier = nn.Linear(128 * INPUT_H * INPUT_W, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


# ── Train / Eval loops ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── Load raw training data (BS3–BS13) ─────────────────────────────────────
    print("Loading training data (BS3–BS13) …")
    train_Xs, train_ys = [], []
    for bs in OUTDOOR_BSS:
        X, y = load_raw(bs, is_outdoor=True)
        train_Xs.append(X)
        train_ys.append(y)
    X_train = np.concatenate(train_Xs, axis=0)  # (N_total, 32)
    y_train = np.concatenate(train_ys, axis=0)

    # ── Compute normalization stats from training data only ───────────────────
    mu  = X_train.mean(axis=0, keepdims=True)          # (1, 32)
    std = X_train.std(axis=0,  keepdims=True) + 1e-8   # (1, 32)

    train_ds     = make_dataset(X_train, y_train, mu, std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    print(f"  Total train samples : {len(train_ds):,}\n")

    # ── Load test data and normalize with train stats ─────────────────────────
    print("Loading test data (BS14–BS15) …")
    test_loaders = {}
    for bs in INDOOR_BSS:
        X_test, y_test = load_raw(bs, is_outdoor=False)
        test_ds = make_dataset(X_test, y_test, mu, std)
        test_loaders[bs] = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                      num_workers=2, pin_memory=True)
        print(f"  BS{bs} samples : {len(test_ds):,}")
    print()

    # ── Model / optimiser / loss ───────────────────────────────────────────────
    model     = BeamNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}")
    print("─" * 32)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % 10 == 0:
            print(f"{epoch:6d}  {tr_loss:10.4f}  {tr_acc*100:9.2f}%")

    # ── Test ───────────────────────────────────────────────────────────────────
    print("\n── Test Results ──────────────────────────────────")
    for bs, loader in test_loaders.items():
        _, acc = evaluate(model, loader, criterion, device)
        print(f"  BS{bs} Top-1 Accuracy : {acc*100:.2f}%")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    torch.save({"model": model.state_dict(), "mu": mu, "std": std},
               "baseline_beamnet.pth")
    print("\nModel saved → baseline_beamnet.pth")


if __name__ == "__main__":
    main()
