"""
Bore labeler — annotate frames in a dataset folder.

Reads from  <dataset>/unlabeled/
Writes to   <dataset>/labeled/  +  labeled/labels.json

Keys / mouse:
  LEFT-CLICK        — place bore dot
  SCROLL            — zoom in / out  (centered on cursor, up to 40x)
  RIGHT-DRAG        — pan
  SPACE / ENTER     - confirm label for current frame -> moves to labeled/, advance
  D / right         - skip frame (keep in unlabeled), advance
  A / left          - go to previous frame
  R                 - reset dot to image center
  Q / ESC           - quit (saves labels.json)

Marker colors:
  YELLOW = click-placed
  CYAN   = default (image center, not manually set)

Usage:
    python labeler_ui.py datasets/cam1_z3.75x/
"""
import cv2, json, os, sys, shutil
import numpy as np

# ── Setup ─────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

DATASET_DIR  = os.path.abspath(sys.argv[1])
UNLABELED    = os.path.join(DATASET_DIR, "unlabeled")
LABELED      = os.path.join(DATASET_DIR, "labeled")
LABELS_FILE  = os.path.join(LABELED, "labels.json")

os.makedirs(LABELED, exist_ok=True)

# Load existing labels (if resuming)
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE) as f:
        labels = json.load(f)
else:
    labels = {}

# Collect frames — only those still in unlabeled/ (not yet labeled)
exts = {".jpg", ".jpeg", ".png"}
frames = sorted(
    f for f in os.listdir(UNLABELED)
    if os.path.splitext(f)[1].lower() in exts
)
if not frames:
    print(f"No images in {UNLABELED}")
    sys.exit(0)

DISPLAY_W = 800
DISPLAY_H = 600

# ── View state ────────────────────────────────────────────────────────────────
view = {
    "zoom":  1.0,
    "pan_x": 0.0,
    "pan_y": 0.0,
    "img_w": 1,
    "img_h": 1,
    "drag":  False,
    "drag_sx": 0, "drag_sy": 0,
    "drag_px": 0.0, "drag_py": 0.0,
}

# Per-frame state: bore position and whether it was manually placed
bore   = {}    # fname → (x, y)
manual = set() # fnames with click-placed dots

def _default_bore(fname):
    p = os.path.join(UNLABELED, fname)
    img = cv2.imread(p)
    if img is None:
        return (128, 128)
    H, W = img.shape[:2]
    return (W // 2, H // 2)

def _clamp_pan(v):
    max_px = max(0.0, v["img_w"] - DISPLAY_W / v["zoom"])
    max_py = max(0.0, v["img_h"] - DISPLAY_H / v["zoom"])
    v["pan_x"] = max(0.0, min(v["pan_x"], max_px))
    v["pan_y"] = max(0.0, min(v["pan_y"], max_py))

def _disp_to_img(dx, dy):
    ix = view["pan_x"] + dx / view["zoom"]
    iy = view["pan_y"] + dy / view["zoom"]
    return int(round(ix)), int(round(iy))

def _img_to_disp(ix, iy):
    dx = (ix - view["pan_x"]) * view["zoom"]
    dy = (iy - view["pan_y"]) * view["zoom"]
    return int(round(dx)), int(round(dy))

def init_view(fname):
    """Fit image to window — call only when switching frames."""
    p = os.path.join(UNLABELED, fname)
    img = cv2.imread(p)
    if img is None:
        return
    H, W = img.shape[:2]
    view["img_w"] = W
    view["img_h"] = H
    view["zoom"]  = min(DISPLAY_W / W, DISPLAY_H / H)
    view["pan_x"] = 0.0
    view["pan_y"] = 0.0

def render(idx):
    fname = frames[idx]
    p = os.path.join(UNLABELED, fname)
    img = cv2.imread(p)
    if img is None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
    H, W = img.shape[:2]
    view["img_w"] = W
    view["img_h"] = H
    _clamp_pan(view)

    z   = view["zoom"]
    px, py = view["pan_x"], view["pan_y"]
    x1, y1 = int(px), int(py)
    x2 = min(W, x1 + int(DISPLAY_W / z) + 1)
    y2 = min(H, y1 + int(DISPLAY_H / z) + 1)
    patch = img[y1:y2, x1:x2]
    out_w = max(1, min(int((x2 - x1) * z), DISPLAY_W))
    out_h = max(1, min(int((y2 - y1) * z), DISPLAY_H))
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    canvas[:out_h, :out_w] = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Bore dot
    bx, by = bore.get(fname, _default_bore(fname))
    dx, dy = _img_to_disp(bx, by)
    color  = (0, 200, 255) if fname in manual else (255, 220, 0)   # yellow=click, cyan=default
    cv2.circle(canvas, (dx, dy), 2, color, -1)

    # HUD
    n_labeled   = len(labels)
    n_remaining = len(frames)
    n_total     = n_labeled + n_remaining
    status = "PLACED" if fname in manual else "default (click to set)"
    line = (f"[{idx+1}/{n_remaining} unlabeled]  "
            f"labeled: {n_labeled}/{n_total}  |  "
            f"bore=({bx},{by})  {status}  zoom={z:.1f}x  "
            f"SPACE=label  D=skip  A=back  Q=quit")
    cv2.putText(canvas, line, (6, DISPLAY_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.imshow("labeler", canvas)

def save_labels():
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)

def commit_label(idx):
    """Move frame from unlabeled/ to labeled/ and record annotation."""
    fname = frames[idx]
    bx, by = bore.get(fname, _default_bore(fname))
    src = os.path.join(UNLABELED, fname)
    dst = os.path.join(LABELED,   fname)
    if os.path.exists(src):
        shutil.move(src, dst)

    # Load image dimensions for normalization
    img = cv2.imread(dst)
    H, W = (img.shape[:2] if img is not None else (256, 256))
    labels[fname] = {
        "bore_x":      bx,
        "bore_y":      by,
        "bore_x_norm": bx / max(W, 1),
        "bore_y_norm": by / max(H, 1),
    }
    save_labels()
    frames.pop(idx)
    bore.pop(fname, None)
    manual.discard(fname)


# ── Window + callbacks ────────────────────────────────────────────────────────
cv2.namedWindow("labeler", cv2.WINDOW_NORMAL)
cv2.resizeWindow("labeler", DISPLAY_W, DISPLAY_H)

idx = 0
init_view(frames[idx])
render(idx)

def on_mouse(event, x, y, flags, _):
    global idx
    z = view["zoom"]

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = _disp_to_img(x, y)
        fname = frames[idx]
        bore[fname]   = (max(0, min(ix, view["img_w"] - 1)),
                         max(0, min(iy, view["img_h"] - 1)))
        manual.add(fname)
        render(idx)

    elif event == cv2.EVENT_RBUTTONDOWN:
        view["drag"] = True
        view["drag_sx"] = x;  view["drag_sy"] = y
        view["drag_px"] = view["pan_x"]
        view["drag_py"] = view["pan_y"]

    elif event == cv2.EVENT_MOUSEMOVE and view["drag"]:
        dx = (x - view["drag_sx"]) / z
        dy = (y - view["drag_sy"]) / z
        view["pan_x"] = view["drag_px"] - dx
        view["pan_y"] = view["drag_py"] - dy
        _clamp_pan(view)
        render(idx)

    elif event == cv2.EVENT_RBUTTONUP:
        view["drag"] = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        factor = 1.25 if flags > 0 else 0.8
        old_z  = z
        new_z  = max(0.5, min(40.0, old_z * factor))
        img_x  = view["pan_x"] + x / old_z
        img_y  = view["pan_y"] + y / old_z
        view["zoom"]  = new_z
        view["pan_x"] = img_x - x / new_z
        view["pan_y"] = img_y - y / new_z
        _clamp_pan(view)
        render(idx)

cv2.setMouseCallback("labeler", on_mouse)

while True:
    key = cv2.waitKey(0) & 0xFF

    if key in (ord('q'), 27):                          # quit
        save_labels()
        print(f"Saved {len(labels)} labeled frames → {LABELS_FILE}")
        break

    elif key in (13, 32):                              # ENTER or SPACE = label
        if not frames:
            break
        commit_label(idx)
        if not frames:
            print("All frames labeled!")
            break
        idx = min(idx, len(frames) - 1)
        init_view(frames[idx])
        render(idx)

    elif key in (ord('d'), 83):                        # D or → = skip
        if frames:
            idx = min(idx + 1, len(frames) - 1)
            init_view(frames[idx])
            render(idx)

    elif key in (ord('a'), 81):                        # A or ← = back
        idx = max(idx - 1, 0)
        init_view(frames[idx])
        render(idx)

    elif key == ord('r'):                              # reset dot to center
        fname = frames[idx]
        bore.pop(fname, None)
        manual.discard(fname)
        render(idx)

cv2.destroyAllWindows()
