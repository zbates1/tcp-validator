"""
Train a small bore-localization CNN.

Input : 128x128 crop centered on luer cap (from extract_labels.py)
Output: (x_norm, y_norm) in [0,1] — bore center relative to raw crop size

Usage:
    conda run -n ml311 python train_cnn.py
    conda run -n ml311 python train_cnn.py --epochs 100 --batch 32

Saves: bore_cnn.pt  (TorchScript, loadable without the class definition)
"""
import argparse, json, os, random, sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_data")
SAVE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bore_cnn.pt")
CROP_SIZE  = 128
VAL_FRAC   = 0.15
SEED       = 42


# ── Dataset ────────────────────────────────────────────────────────────────────
class BoreDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self): return len(self.records)

    def _load_img(self, rec):
        path = os.path.join(DATA_DIR, rec["file"])  # 128x128 crop
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None, None, None

        # Bore position in raw crop → normalize to [0,1] via raw image size
        raw_path = os.path.join(DATA_DIR, rec.get("file_raw", rec["file"]))
        try:
            raw_img = Image.open(raw_path)
            rW, rH  = raw_img.size
        except Exception:
            rH, rW = img.shape[:2]
        bx_norm  = rec["bore_x"] / max(rW, 1)
        by_norm  = rec["bore_y"] / max(rH, 1)
        return img, float(bx_norm), float(by_norm)

    def __getitem__(self, idx):
        img, bx, by = self._load_img(self.records[idx])
        if img is None:
            img = np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
            bx, by = 0.5, 0.5

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C H W

        if self.augment:
            img, bx, by = self._augment(img, bx, by)

        # Normalize image to [-1, 1]
        img = img * 2.0 - 1.0
        return img, torch.tensor([bx, by], dtype=torch.float32)

    def _augment(self, img, bx, by):
        # img is C×H×W float [0,1] tensor
        # Horizontal flip
        if random.random() < 0.5:
            img = img.flip(-1)
            bx  = 1.0 - bx
        # Vertical flip
        if random.random() < 0.5:
            img = img.flip(-2)
            by  = 1.0 - by
        # Rotation ±20° via PIL round-trip
        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            arr   = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil   = Image.fromarray(arr).rotate(angle, resample=Image.BILINEAR)
            img   = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
            # Rotate bore coords around center (0.5, 0.5)
            rad   = -angle * np.pi / 180
            dx, dy = bx - 0.5, by - 0.5
            bx    = 0.5 + dx * np.cos(rad) - dy * np.sin(rad)
            by    = 0.5 + dx * np.sin(rad) + dy * np.cos(rad)
            bx, by = float(np.clip(bx, 0, 1)), float(np.clip(by, 0, 1))
        # Brightness / contrast jitter (channel-wise scale + shift)
        if random.random() < 0.7:
            scale = random.uniform(0.6, 1.4)
            shift = random.uniform(-0.1, 0.1)
            img   = (img * scale + shift).clamp(0, 1)
        return img, bx, by


# ── Model ──────────────────────────────────────────────────────────────────────
def conv_bn(ci, co, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, k, s, p, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )

class BoreCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            conv_bn(3, 32),   nn.MaxPool2d(2),   # 64
            conv_bn(32, 64),  nn.MaxPool2d(2),   # 32
            conv_bn(64, 128), nn.MaxPool2d(2),   # 16
            conv_bn(128, 128), nn.MaxPool2d(2),  # 8
            conv_bn(128, 64),                    # 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Sigmoid(),   # output in [0,1]
        )

    def forward(self, x): return self.head(self.enc(x))


# ── Training ───────────────────────────────────────────────────────────────────
def pixel_error(pred, tgt):
    """Mean pixel error in 128px space."""
    return ((pred - tgt).abs().mean() * CROP_SIZE).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",  type=int, default=80)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(os.path.join(DATA_DIR, "labels.json")) as f:
        records = json.load(f)

    # Train / val split (shuffle, keep video balance)
    random.shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRAC))
    val_recs  = records[:n_val]
    train_recs = records[n_val:]
    print(f"Train: {len(train_recs)}   Val: {len(val_recs)}")

    train_ds = BoreDataset(train_recs, augment=True)
    val_ds   = BoreDataset(val_recs,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers)

    model = BoreCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.MSELoss()

    best_val = float("inf")
    patience = 15; no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        t_loss = t_px = 0
        for imgs, tgts in train_dl:
            imgs, tgts = imgs.to(device), tgts.to(device)
            pred = model(imgs)
            loss = crit(pred, tgts)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item()
            t_px   += pixel_error(pred.detach().cpu(), tgts.cpu())
        sched.step()

        # ── val ──
        model.eval()
        v_loss = v_px = 0
        with torch.no_grad():
            for imgs, tgts in val_dl:
                imgs, tgts = imgs.to(device), tgts.to(device)
                pred = model(imgs)
                v_loss += crit(pred, tgts).item()
                v_px   += pixel_error(pred.cpu(), tgts.cpu())

        t_loss /= len(train_dl); v_loss /= len(val_dl)
        t_px   /= len(train_dl); v_px   /= len(val_dl)

        improved = "★" if v_loss < best_val else ""
        print(f"Ep {epoch:03d}  train mse={t_loss:.5f} px={t_px:.1f}  "
              f"val mse={v_loss:.5f} px={v_px:.1f}  {improved}")

        if v_loss < best_val:
            best_val = v_loss; no_improve = 0
            # Save as TorchScript so it loads without BoreCNN class
            scripted = torch.jit.script(model.cpu())
            scripted.save(SAVE_PATH)
            model.to(device)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\nBest val mse={best_val:.5f}  →  {SAVE_PATH}")


if __name__ == "__main__":
    main()
