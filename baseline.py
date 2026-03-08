"""
Baseline: Beam Selection via CNN
Train : BS3–BS13 (outdoor, top-50k users per BS)
Test  : BS14–BS15 (indoor, all users)

Input  : R^32  (received powers from 32 wide beams)
Label  : {0,1}^128 one-hot  →  class index 0–127
"""

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "."          # .mat files directory
OUTDOOR_BSS  = list(range(3, 14))   # BS3–BS13  (train)
INDOOR_BSS   = [14, 15]             # BS14–BS15 (test)
TOP_K        = 50_000               # keep top-K users for outdoor BSs
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
NUM_CLASSES  = 128
INPUT_H, INPUT_W = 4, 8  # reshape 32 → (1, 4, 8)

# ── .mat key names ────────────────────────────────────────────────────────────
KEY_INPUT = "X"        # shape (N, 32)
KEY_LABEL = "best128"  # shape (N, 1) or (N,) — integer beam index (1-indexed)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BeamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, 1, 4, 8)  y: (N,) int64
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_bs(bs_idx: int, is_outdoor: bool) -> BeamDataset:
    path = os.path.join(DATA_DIR, f"beam_dataset_bs{bs_idx}.mat")
    data = sio.loadmat(path)

    X = data[KEY_INPUT].astype(np.float32)   # (N, 32)
    y = data[KEY_LABEL].astype(np.float32)   # (N, 128) one-hot

    if is_outdoor:
        # keep top-50k by maximum received power
        max_power = X.max(axis=1)
        top_idx   = np.argsort(max_power)[-TOP_K:]
        X, y      = X[top_idx], y[top_idx]

    labels = y.flatten().astype(np.int64) - 1          # 1-indexed → 0-indexed
    X      = X.reshape(-1, 1, INPUT_H, INPUT_W)        # (N, 1, 4, 8)
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
        x = self.features(x)        # (B, 128, 4, 8)
        x = x.flatten(start_dim=1)  # (B, 4096)
        return self.classifier(x)   # (B, 128) — logits


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

    # ── Build train set (BS3–BS13, outdoor, top-50k each) ─────────────────────
    print("Loading training data (BS3–BS13) …")
    train_sets = [load_bs(bs, is_outdoor=True) for bs in OUTDOOR_BSS]
    train_ds   = ConcatDataset(train_sets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    print(f"  Total train samples : {len(train_ds):,}\n")

    # ── Build test sets (BS14–BS15, indoor, all users) ─────────────────────────
    print("Loading test data (BS14–BS15) …")
    test_loaders = {
        bs: DataLoader(load_bs(bs, is_outdoor=False), batch_size=BATCH_SIZE,
                       num_workers=4, pin_memory=True)
        for bs in INDOOR_BSS
    }
    for bs, loader in test_loaders.items():
        print(f"  BS{bs} samples : {len(loader.dataset):,}")
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
    torch.save(model.state_dict(), "baseline_beamnet.pth")
    print("\nModel saved → baseline_beamnet.pth")


if __name__ == "__main__":
    main()
