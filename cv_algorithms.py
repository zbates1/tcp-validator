#!/usr/bin/env python3
import cv2, numpy as np, os

# ── CV tuning constants ───────────────────────────────────────────────────────
# Saturation-blob params for cap detection (color-agnostic)
CAP_SAT_MIN  = 75      # HSV S threshold — cap (any color) is always more saturated than white barrel or gray arm
CAP_AREA_MIN = 5000    # px² — ignore tiny specular dots; real cap is always larger

# Bore = darkest compact circular blob (1.3px median vs hand labels on xy_cam_3x)
BORE_V_MAX    = 70     # HSV V threshold — bore is near-black against the cap
BORE_AREA_MIN = 300    # px²
BORE_AREA_MAX = 20000  # px²
BORE_FILL_MIN = 0.6    # contour_area / enclosing_circle_area — rejects ragged shadows

SMOOTH_ALPHA = 0.45    # EMA weight on new detection (lower = smoother but laggier)

# Lucas-Kanade optical flow params
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
LK_FB_THRESH       = 8.0   # px — max forward-backward error before re-acquire
# frames between full re-detects; 0 = detect every frame (LK off). On the 10.8.26
# xy video LK drifted >20px on 12% of frames; per-frame detect is 1px and ~14ms.
LK_REACQUIRE_EVERY = 0

SIDE_LO, SIDE_HI, SIDE_GAP = 50, 150, 40        # Canny / HoughLinesP (side cam)
SIDE_MIN_CONTRAST = 8   # gray levels — needle vs background needed to refine the bottom edge

# Hough circle bore fallback (same params as the notebook comparison: 400px crop scaled to 224px).
# xy_cam_3x labels: 1.8px median with the crop off-center by up to 60px (dark-blob 1.0px); found the bore on all of
# 182236, where the dark-blob finds nothing.
HOUGH_CROP, HOUGH_SIZE = 400, 224
HOUGH_PARAM1, HOUGH_PARAM2, HOUGH_MIN_DIST = 60, 20, 30
HOUGH_MIN_R, HOUGH_MAX_R = 8, 60       # px in the 224px image

last_bore_method = None   # which detect_bottom stage produced the last result: blob / hough / dino / min
# ─────────────────────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# ── DINOv2 bore head ─────────────────────────────────────────────────────────
# Fine-tuned regression head on top of frozen DINOv2-small.
# Weights live at models/dinov2_head.pt (copy your best run there to update).
# Trained on 400px crops centered on the bore (±15% jitter), so the crop center
# must already be near the bore — this is a fallback, not a cold-start locator.
_DINO_MODEL    = "facebook/dinov2-small"
_DINO_HEAD_PT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dinov2_head.pt")
_DINO_CROP     = 400   # must match crop_size used in training
_dino_backbone = None
_dino_head     = None
_dino_tf       = None

def _load_dino():
    global _dino_backbone, _dino_head, _dino_tf
    if not os.path.exists(_DINO_HEAD_PT):
        print(f"DINOv2 head not found at {_DINO_HEAD_PT} — geometric-only detection")
        return
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel
        from torchvision import transforms

        # Cache first: a network check stalls ~25s when Avast blocks huggingface.co
        try:
            backbone = AutoModel.from_pretrained(_DINO_MODEL, local_files_only=True)
        except OSError:
            backbone = AutoModel.from_pretrained(_DINO_MODEL)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        feat_dim = backbone.config.hidden_size  # 384 for dinov2-small

        class _BoreHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(feat_dim, 128), nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 2),
                    nn.Sigmoid(),
                )
            def forward(self, x): return self.net(x)

        head = _BoreHead()
        head.load_state_dict(torch.load(_DINO_HEAD_PT, map_location="cpu"))
        head.eval()

        _dino_backbone = backbone
        _dino_head     = head
        _dino_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"DINOv2 bore head loaded ({_DINO_MODEL}): {_DINO_HEAD_PT}")
    except Exception as e:
        print(f"DINOv2 load failed ({e}) — geometric-only detection")

def _dino_detect_bore(frame, cx, cy):
    """Crop 400px around (cx, cy), run DINOv2 head. Returns (bx, by) in full-frame coords."""
    if _dino_backbone is None:
        return None
    import torch
    from PIL import Image

    H, W = frame.shape[:2]
    half = _DINO_CROP // 2
    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(W, x1 + _DINO_CROP)
    y2 = min(H, y1 + _DINO_CROP)
    x1 = max(0, x2 - _DINO_CROP)   # re-anchor if clipped at right/bottom edge
    y1 = max(0, y2 - _DINO_CROP)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    pil_img = Image.fromarray(crop[:, :, ::-1].copy())   # BGR -> RGB
    inp = _dino_tf(pil_img).unsqueeze(0)

    with torch.no_grad():
        feats = _dino_backbone(pixel_values=inp).last_hidden_state[:, 0, :]
        pred  = _dino_head(feats)[0]

    cH, cW = crop.shape[:2]
    bx = int(round(float(pred[0]) * cW)) + x1
    by = int(round(float(pred[1]) * cH)) + y1
    return (bx, by)

_load_dino()   # attempt load at import time
# ─────────────────────────────────────────────────────────────────────────────


def crosshair_score(frame):
    """How much greener the center row + column are than their neighbours; high = burned-in Pi crosshair."""
    h, w = frame.shape[:2]
    f = frame.astype(np.int16)
    green = f[:, :, 1] - (f[:, :, 0] + f[:, :, 2]) // 2
    row = green[h // 2, :].mean() - (green[h // 2 - 4, :].mean() + green[h // 2 + 4, :].mean()) / 2
    col = green[:, w // 2].mean() - (green[:, w // 2 - 4].mean() + green[:, w // 2 + 4].mean()) / 2
    return float(min(row, col))


def strip_crosshair(frame):
    """Erase the green crosshair pi_stream_server.py burns into the live stream (center row + column, r=6 ring)
    by interpolating across it. Recordings are written before it's drawn, so offline videos don't need this."""
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    out[cy - 2:cy + 3, :] = np.linspace(out[cy - 3, :].astype(np.float32),
                                        out[cy + 3, :].astype(np.float32), 7)[1:-1].astype(np.uint8)
    out[:, cx - 2:cx + 3] = np.linspace(out[:, cx - 3].astype(np.float32),
                                        out[:, cx + 3].astype(np.float32), 7)[1:-1].transpose(1, 0, 2).astype(np.uint8)
    roi = out[cy - 12:cy + 13, cx - 12:cx + 13]
    ring = np.zeros(roi.shape[:2], np.uint8)
    cv2.circle(ring, (12, 12), 6, 255, 4)
    roi[:] = cv2.inpaint(roi, ring, 3, cv2.INPAINT_TELEA)
    return out


def track_bore_lk(prev_gray, curr_gray, pt):
    """Lucas-Kanade optical flow with forward-backward consistency check.
    Returns new (x, y) float32 array, or None if tracking failed."""
    p0 = np.float32(pt).reshape(1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK_PARAMS)
    if st is None or st[0, 0] == 0:
        return None
    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **LK_PARAMS)
    if st2 is None or st2[0, 0] == 0:
        return None
    if np.linalg.norm(p0_back[0, 0] - p0[0, 0]) > LK_FB_THRESH:
        return None
    return p1[0, 0]   # shape (2,) float32


def _hough_bore(frame, center=None):
    """Hough circle on a HOUGH_CROP crop around `center` (the circle nearest it), or on the whole frame at the same
    scale (the circle with the darkest inside) when there's no previous position. Returns (x, y, r) or None."""
    H, W = frame.shape[:2]
    s = HOUGH_SIZE / HOUGH_CROP
    crop = center is not None and W >= HOUGH_CROP and H >= HOUGH_CROP
    if crop:
        x1 = int(min(max(0, center[0] - HOUGH_CROP // 2), W - HOUGH_CROP))
        y1 = int(min(max(0, center[1] - HOUGH_CROP // 2), H - HOUGH_CROP))
        img = frame[y1:y1 + HOUGH_CROP, x1:x1 + HOUGH_CROP]
    else:
        x1 = y1 = 0
        img = frame
    small = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=HOUGH_MIN_DIST, param1=HOUGH_PARAM1,
                         param2=HOUGH_PARAM2, minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R)
    if c is None:
        return None
    c = c.reshape(-1, 3)   # OpenCV 4 returns (1,N,3), OpenCV 5 (N,3)
    if crop:
        tx, ty = (center[0] - x1) * s, (center[1] - y1) * s
        best = min(c, key=lambda k: (k[0] - tx) ** 2 + (k[1] - ty) ** 2)
    else:
        def _inside(k):
            m = np.zeros_like(g)
            cv2.circle(m, (int(k[0]), int(k[1])), max(1, int(k[2] * 0.6)), 255, -1)
            return cv2.mean(g, m)[0]
        best = min(c, key=_inside)
    return x1 + float(best[0]) / s, y1 + float(best[1]) / s, float(best[2]) / s


def _hough_result(frame, last_tip, circles):
    global last_bore_method
    h = _hough_bore(frame, last_tip)
    if h is None:
        return None
    last_bore_method = "hough"
    return (int(round(h[0])), int(round(h[1]))), circles, None


def detect_bottom(frame, last_tip=None, max_jump=200):
    """Stage 1: cap = saturated circular blob. Stage 2: bore = dark circular blob.
    Fallbacks: Hough circle (cropped around the last bore), DINOv2 head, then V+S minimum inside the cap.
    Sets last_bore_method to the stage that answered."""
    global last_bore_method
    last_bore_method = None
    H, W = frame.shape[:2]

    # ── Stage 1: locate luer cap via saturation + circularity ────────────────
    # The cap (any color) is a saturated circular blob.
    # Wood/background may also be saturated but is not circular.
    # Fill ratio = contour_area / enclosing_circle_area → ~0.7 for a circle,
    # much lower for irregular shapes.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv[:, :, 1], CAP_SAT_MIN, 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN,  k)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in contours if cv2.contourArea(c) >= CAP_AREA_MIN]
    if not cs:
        return _hough_result(frame, last_tip, []) or (None, [], None)

    def _fill_ratio(c):
        area = cv2.contourArea(c)
        (_, _), r = cv2.minEnclosingCircle(c)
        return area / (np.pi * r * r) if r > 0 else 0.0

    # Score = fill_ratio; prefer the most circular blob.
    # When tracking, prefer blobs near last known position.
    if last_tip is not None:
        def _d(c):
            (x, y), _ = cv2.minEnclosingCircle(c)
            return ((x - last_tip[0])**2 + (y - last_tip[1])**2)**0.5
        nearby = [c for c in cs if _d(c) < max_jump]
        candidates = nearby if nearby else cs
        cap = max(candidates, key=lambda c: _fill_ratio(c) - 0.1 * (_d(c) / max_jump))
    else:
        cap = max(cs, key=_fill_ratio)

    (cap_cx, cap_cy), cap_r = cv2.minEnclosingCircle(cap)
    cap_cx, cap_cy, cap_r = int(cap_cx), int(cap_cy), int(cap_r)

    # ── Stage 2: bore = darkest compact circular blob ────────────────────────
    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    dark = cv2.inRange(v, 0, BORE_V_MAX)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts_d, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in cnts_d
             if BORE_AREA_MIN < cv2.contourArea(c) < BORE_AREA_MAX
             and _fill_ratio(c) > BORE_FILL_MIN]
    if blobs:
        if last_tip is not None:
            def _dist_to_last(c):
                (bx, by), _ = cv2.minEnclosingCircle(c)
                return (bx - last_tip[0]) ** 2 + (by - last_tip[1]) ** 2
            best = min(blobs, key=_dist_to_last)
        else:
            best = max(blobs, key=_fill_ratio)   # cold start: most circular
        M = cv2.moments(best)
        bx, by = int(round(M["m10"] / M["m00"])), int(round(M["m01"] / M["m00"]))
        (ecx, ecy), er = cv2.minEnclosingCircle(best)
        last_bore_method = "blob"
        return (bx, by), [(cap_cx, cap_cy, cap_r)], (int(ecx), int(ecy), int(er))

    # Fallback: Hough circle, cropped around the last known bore (whole frame on a cold start)
    hough = _hough_result(frame, last_tip, [(cap_cx, cap_cy, cap_r)])
    if hough is not None:
        return hough

    # Fallback: DINOv2 head, cropped around the last known bore (else cap center)
    ccx, ccy = last_tip if last_tip is not None else (cap_cx, cap_cy)
    dino_tip = _dino_detect_bore(frame, ccx, ccy)
    if dino_tip is not None:
        last_bore_method = "dino"
        return dino_tip, [(cap_cx, cap_cy, cap_r)], None

    # Last resort: V+S minimum within filled cap contour (no model available)
    cap_fill = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(cap_fill, [cap], -1, 255, -1)
    inner_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(inner_mask, (cap_cx, cap_cy), max(1, int(cap_r * 0.42)), 255, -1)
    inner_mask = cv2.bitwise_and(inner_mask, cap_fill)
    v_f = hsv[:, :, 2].astype(np.float32)
    s_f = hsv[:, :, 1].astype(np.float32)
    vs = v_f + s_f
    vs[(inner_mask == 0) | (v_f > 160)] = 512.0
    score_map = cv2.GaussianBlur(vs, (7, 7), 2)
    score_map[inner_mask == 0] = 512.0
    min_idx = np.unravel_index(score_map.argmin(), score_map.shape)
    last_bore_method = "min"
    return (int(min_idx[1]), int(min_idx[0])), [(cap_cx, cap_cy, cap_r)], None


def _needle_bottom(gray, vert, tip, win=40):
    """Sub-pixel bottom end of the needle, or None to keep the Hough endpoint.
    The segments only give the needle's direction (often only one wall is found). Its center line and width come from
    a cross-section just above the tip; the end is where the needle's difference from the background beside it drops
    to half, searched from above the endpoint to 2*win below it (endpoints often stop short). 10.19.26 side videos:
    90th pct frame-to-frame jitter 0.1-0.5px in focus vs 1-6px for the raw endpoint, which sat up to 60px too high."""
    pts = np.float32([(x, y) for x1, y1, x2, y2 in vert for x, y in ((x1, y1), (x2, y2))])
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if vy < 0:
        vx, vy = -vx, -vy                                # axis points down the needle
    nx, ny = vy, -vx
    t_tip = (tip[0] - x0) * vx + (tip[1] - y0) * vy

    def sample(ts, offs):
        X = (x0 + ts[:, None] * vx + offs[None, :] * nx).astype(np.float32)
        Y = (y0 + ts[:, None] * vy + offs[None, :] * ny).astype(np.float32)
        return cv2.remap(gray, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Cross-section above the tip -> needle center line and half width
    offs = np.arange(-30, 30.5, 1.0)
    P = sample(np.arange(t_tip - 40, t_tip - 8, 2.0), offs)
    dev = np.abs(P - np.median(np.concatenate([P[:, :6], P[:, -6:]], axis=1), axis=1, keepdims=True)).mean(axis=0)
    floor, peak = np.median(np.r_[dev[:6], dev[-6:]]), dev.max()
    if peak - floor < SIDE_MIN_CONTRAST:
        return None
    on = dev > floor + 0.5 * (peak - floor)
    l = r = int(dev.argmax())
    while l > 0 and on[l - 1]: l -= 1
    while r < len(on) - 1 and on[r + 1]: r += 1
    off, half_w = (offs[l] + offs[r]) / 2, (offs[r] - offs[l]) / 2 + 0.5
    if not 2 <= half_w <= 22:
        return None
    x0, y0 = x0 + off * nx, y0 + off * ny

    # Along the axis: needle core vs the background beside it (abs, so a bright reflection stripe counts as needle)
    ts = np.arange(t_tip - win, t_tip + 2 * win + 0.25, 0.5)
    core = sample(ts, np.linspace(-0.6 * half_w, 0.6 * half_w, 7))
    side = sample(ts, np.r_[np.linspace(-half_w - 12, -half_w - 5, 4), np.linspace(half_w + 5, half_w + 12, 4)])
    con = np.abs(core - np.median(side, axis=1, keepdims=True)).mean(axis=1)
    inside, outside = np.median(con[ts < t_tip - 12]), np.percentile(con, 10)
    if inside - outside < SIDE_MIN_CONTRAST:
        return None
    a = con - (inside + outside) / 2                     # > 0 on the needle, < 0 past its end
    stay = 24                                            # 12px of background must follow the end
    for i in range(int(np.searchsorted(ts, t_tip - 12)), len(a) - stay):
        if a[i] > 0 and (a[i + 1:i + 1 + stay] <= 0).all():
            t_b = ts[i] + 0.5 * a[i] / (a[i] - a[i + 1])
            return float(x0 + t_b * vx), float(y0 + t_b * vy)
    return None


def detect_side(frame, tip_at_bottom=True):
    """Side-view needle tip detection (from needle_detection.ipynb). Returns ((x, y) float or None, segments).

    Primary: Hough lines on Canny edges — finds near-vertical line segments
    (needle shaft); the bottommost (or topmost) endpoint is the rough tip, then
    the bottom is refined to sub-pixel along the needle (_needle_bottom).

    Fallback: contour thresholding — brightest-against-dark blob with high
    aspect ratio; used if Hough lines find nothing.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Hough lines (primary) ──────────────────────────────────────────────
    edges = cv2.Canny(gray, SIDE_LO, SIDE_HI)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                             minLineLength=50, maxLineGap=SIDE_GAP)
    if lines is not None:
        # reshape: OpenCV 4 returns (N,1,4), OpenCV 5 returns (N,4)
        vert = [tuple(int(v) for v in l) for l in lines.reshape(-1, 4)
                if abs(l[2] - l[0]) < abs(l[3] - l[1])]
        if vert:
            pts = [(x, y) for x1, y1, x2, y2 in vert
                   for x, y in [(x1, y1), (x2, y2)]]
            tip = (max(pts, key=lambda p: p[1]) if tip_at_bottom
                   else min(pts, key=lambda p: p[1]))
            if tip_at_bottom:
                g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
                tip = _needle_bottom(g, vert, tip) or tip
            return (float(tip[0]), float(tip[1])), vert

    # ── Contour fallback ───────────────────────────────────────────────────
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def _aspect(c):
        x, y, w, h = cv2.boundingRect(c)
        return max(w, h) / max(min(w, h), 1)

    candidates = [c for c in cnts if _aspect(c) > 5]
    if not candidates:
        return None, []
    needle = max(candidates, key=cv2.contourArea)
    idx    = needle[:, :, 1].argmax() if tip_at_bottom else needle[:, :, 1].argmin()
    return (float(needle[idx][0][0]), float(needle[idx][0][1])), []


def draw_overlay(frame, tip, cx, cy, err, vx, vy, armed, circles=None,
                 lk_mode=False, bore_circle=None, mode=0, method=None):
    """mode 0 = center dot, mode 1 = bore circle (radius view). method = detect_bottom stage, shown next to err."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (0, cy), (w, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, 0), (cx, h), (0, 255, 0), 1)
    cv2.circle(out, (cx, cy), 6, (0, 255, 0), 1)
    for c in (circles or []):
        cv2.circle(out, (c[0], c[1]), c[2], (100, 100, 100), 1)

    if tip:
        if mode == 1:
            if bore_circle is not None:
                bcx, bcy, br = bore_circle
            else:
                bcx, bcy, br = tip[0], tip[1], 20   # fallback circle when hole not found
            cv2.circle(out, (bcx, bcy), br, (0, 100, 255), 2)
            cv2.circle(out, (bcx, bcy), 4, (0, 100, 255), -1)
            cv2.line(out, (bcx, bcy), (bcx + br, bcy), (0, 100, 255), 1)
        else:
            cv2.circle(out, tip, 8, (0, 0, 255), -1)
        cv2.line(out, tip, (cx, cy), (255, 0, 255), 1)

    mode_label = "RADIUS" if mode == 1 else "CENTER"
    if tip and mode == 1 and bore_circle:
        det_str = f"err {err:.1f}px  r={bore_circle[2]}px"
    else:
        det_str = f"err {err:.1f}px" if tip else "NO DETECTION"
    if tip and method:
        det_str += f"  ({method})"

    texts = [
        (det_str,                                          (0, 200, 255)),
        (f"vx={vx*1e3:.1f}  vy={vy*1e3:.1f} mm/s",        (200, 200, 200)),
        ("z=align Z  c=cal XY  SPACE=center XY  wasd/rf=jog  q=quit",  (180, 180, 0)),
        (">>> MOVING <<<" if armed else "--- PAUSED ---",  (0, 80, 255) if armed else (100, 100, 100)),
    ]
    for i, (txt, col) in enumerate(texts):
        cv2.putText(out, txt, (10, 34 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

    # Mode badge — top-right corner
    badge     = f" XY: {mode_label} "
    badge_col = (0, 100, 255) if mode == 1 else (0, 200, 80)
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    bx = w - bw - 8
    cv2.rectangle(out, (bx - 4, 6), (w - 4, bh + 14), badge_col, -1)
    cv2.putText(out, badge, (bx, bh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def draw_side_overlay(frame, tip, side_lines=None, mode=0, z_band=None):
    """YZ side-view overlay. Only Z is servoed from this view, so the target is the horizontal center line (green);
    the magenta line runs through the needle's bottom and turns green once it is within z_band px of the target.
    mode 0 = bottom line only
    mode 1 = also the detected Hough segments
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cy = h // 2
    cv2.line(out, (0, cy), (w, cy), (0, 255, 0), 2)

    if mode == 1 and side_lines:
        for x1, y1, x2, y2 in side_lines:
            cv2.line(out, (x1, y1), (x2, y2), (0, 220, 255), 2)

    if tip:
        ev  = cy - tip[1]                                 # rows from the bottom to the target (+ = needle too high)
        on  = z_band is not None and abs(ev) < z_band
        col = (0, 255, 0) if on else (255, 0, 255)
        tx, y16 = int(round(tip[0])), int(round(tip[1] * 16))
        cv2.line(out, (0, y16), (w * 16, y16), col, 1, cv2.LINE_AA, shift=4)   # sub-pixel row
        if abs(ev) > 6:
            cv2.arrowedLine(out, (tx, int(round(tip[1]))), (tx, cy), col, 2, tipLength=min(0.3, 12 / abs(ev)))
        cv2.circle(out, (tx, int(round(tip[1]))), 5, col, -1)
        cv2.putText(out, f"Z err {ev:+.1f}px" + ("  ON LINE" if on else ""), (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if on else (0, 200, 255), 2)
    else:
        cv2.putText(out, "YZ NO DETECTION", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

    # Mode badge — top-right corner
    mode_label = "EDGE" if mode == 1 else "TIP"
    badge     = f" YZ: {mode_label} "
    badge_col = (0, 100, 255) if mode == 1 else (0, 200, 80)
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    bx = w - bw - 8
    cv2.rectangle(out, (bx - 4, 6), (w - 4, bh + 14), badge_col, -1)
    cv2.putText(out, badge, (bx, bh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
