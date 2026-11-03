"""
BoreDataset — PyTorch Dataset for bore/needle-tip keypoint data.

Dataset folder structure:
    datasets/
        cam1_z3.75x/
            unlabeled/    frame_00000.jpg ...
            labeled/
                labels.json   { "frame_00000.jpg": {"bore_x": 45, "bore_y": 67, ...} }
                frame_00000.jpg ...

Usage:
    from dataset import BoreDataset

    # Full frame (slow, bore is tiny in large images)
    ds = BoreDataset("datasets", datasets=["cam1_z3.75x"])

    # Crop a 400x400 window around bore — bore is large, task is easy
    ds = BoreDataset("datasets", datasets=["cam1_z3.75x"], crop_size=400)

    # With torchvision transforms
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds = BoreDataset("datasets", crop_size=400, transform=tf)
"""
import json, random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BoreDataset(Dataset):
    """
    Loads bore-center keypoint labels from datasets/<name>/labeled/.

    With crop_size set: returns a fixed-size crop centered on the bore so the
    bore is always a prominent feature regardless of full-frame resolution.
    Returns (PIL.Image, torch.Tensor([x_norm, y_norm])) normalized to [0, 1]
    within whatever region is returned (full image or crop).
    """

    def __init__(self, root="datasets", datasets=None, transform=None,
                 crop_size=None, jitter=0.15, hflip=False):
        """
        root      : path to datasets/ directory
        datasets  : list of subfolder names, or None for all
        transform : optional callable applied to PIL images
        crop_size : if set, crop a square of this size centered on the bore.
                    Bore will be near (0.5, 0.5) in the cropped image.
        jitter    : random offset as fraction of crop_size (train augmentation)
        hflip     : if True, randomly flip image AND mirror bore_x — must be
                    handled here (not in torchvision) so the target stays consistent
        """
        root = Path(root)
        self.transform  = transform
        self.crop_size  = crop_size
        self.jitter     = jitter
        self.hflip      = hflip
        self.samples    = []  # (img_path, bore_x_px, bore_y_px)

        subdirs = ([root / name for name in datasets]
                   if datasets is not None
                   else [p for p in sorted(root.iterdir()) if p.is_dir()])

        for subdir in subdirs:
            labels_path = subdir / "labeled" / "labels.json"
            if not labels_path.exists():
                continue
            with open(labels_path) as f:
                labels = json.load(f)
            for fname, ann in labels.items():
                img_path = subdir / "labeled" / fname
                if img_path.exists():
                    self.samples.append((img_path, ann["bore_x"], ann["bore_y"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bx_px, by_px = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        if self.crop_size is not None:
            half = self.crop_size // 2
            # Random jitter so the bore isn't always exactly centered
            jit = int(self.jitter * self.crop_size)
            ox = random.randint(-jit, jit)
            oy = random.randint(-jit, jit)
            cx = int(bx_px) + ox
            cy = int(by_px) + oy
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(W, x1 + self.crop_size)
            y2 = min(H, y1 + self.crop_size)
            x1 = x2 - self.crop_size  # re-anchor if clipped
            y1 = y2 - self.crop_size
            x1, y1 = max(0, x1), max(0, y1)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            bx_norm = (bx_px - x1) / self.crop_size
            by_norm = (by_px - y1) / self.crop_size
        else:
            bx_norm = bx_px / max(W, 1)
            by_norm = by_px / max(H, 1)

        if self.hflip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            bx_norm = 1.0 - bx_norm   # mirror x, y unchanged

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([bx_norm, by_norm], dtype=torch.float32)

    def summary(self):
        from collections import Counter
        counts = Counter(str(p.parent.parent.name) for p, *_ in self.samples)
        mode = f"crop={self.crop_size}px" if self.crop_size else "full frame"
        print(f"BoreDataset  total={len(self.samples)}  ({mode})")
        for name, n in sorted(counts.items()):
            print(f"  {name}: {n}")
