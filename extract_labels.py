"""
Extract labeled training crops for bore CNN.

For each video, samples ~200 frames, runs detect_bottom to get the cap
center and bore position, then saves:
  - crop_NNNN_TAG_fFFFFF.jpg  : 128x128 (CNN input)
  - raw_NNNN_TAG_fFFFFF.jpg   : full-res cap crop (for labeler display)
  - labels.json               : list of {file, file_raw, bore_x, bore_y,
                                          cx, cy, r, vid, frame}

Usage:
    python extract_labels.py                          # process built-in videos
    python extract_labels.py path/to/vid.mp4 ...      # process arbitrary videos
    python extract_labels.py path/to/vid.mp4 --show   # show detection overlay
    python extract_labels.py path/to/vid.mp4 --n 300  # 300 frames per video
"""
import argparse, cv2, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_algorithms import detect_bottom

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "cnn_data")

# ── Default built-in videos (relative to BASE) ────────────────────────────────
DEFAULT_VIDEOS = [
    (os.path.join(BASE, "raw_cam_footage.mp4"),         "old"),
    (os.path.join(BASE, "raw_cam_footage_high_res.mp4"), "hires"),
    (os.path.join(BASE, "xy_cam_optimized.mp4"),         "opt"),
]

CROP_SIZE = 128

def _tag_from_path(p):
    """Make a short tag from a video filename, e.g. 'capture_cam1' → 'cam1'."""
    base = os.path.splitext(os.path.basename(p))[0]
    # keep last 12 chars to avoid super-long tags
    return base[-12:].replace(" ", "_")


def extract(video_path, tag, frames_per_vid, show, out_dir, records, global_i):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"SKIP (cannot open): {video_path}")
        return global_i

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // frames_per_vid)
    print(f"\n{os.path.basename(video_path)}  ({total} frames, step={step}, tag={tag})")

    prev_tip = None
    saved    = 0
    H = W = None

    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if H is None:
            H, W = frame.shape[:2]

        tip, circles, _ = detect_bottom(frame, prev_tip)
        if tip is None or not circles:
            continue
        prev_tip = tip

        cx, cy, r = circles[0]
        r = int(r)

        x1 = max(0, cx - r); x2 = min(W, cx + r)
        y1 = max(0, cy - r); y2 = min(H, cy + r)
        crop_raw = frame[y1:y2, x1:x2].copy()
        if crop_raw.size == 0:
            continue
        crop128 = cv2.resize(crop_raw, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

        bore_cx_raw = tip[0] - x1
        bore_cy_raw = tip[1] - y1

        fname     = f"crop_{global_i:04d}_{tag}_f{idx:05d}.jpg"
        fname_raw = f"raw_{global_i:04d}_{tag}_f{idx:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname),     crop128)
        cv2.imwrite(os.path.join(out_dir, fname_raw), crop_raw)

        records.append({
            "file":     fname,
            "file_raw": fname_raw,
            "bore_x":   bore_cx_raw,
            "bore_y":   bore_cy_raw,
            "cx": cx, "cy": cy, "r": r,
            "vid": tag,
            "frame": idx,
        })
        global_i += 1
        saved += 1

        if show:
            vis = frame.copy()
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(vis, tip, 8, (0, 0, 255), -1)
            cv2.imshow("detect", cv2.resize(vis, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    print(f"  Saved {saved} crops")
    return global_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="Video files to process (default: built-in list)")
    ap.add_argument("--show", action="store_true", help="Show detection overlay while processing")
    ap.add_argument("--n",    type=int, default=200, help="Frames to sample per video (default: 200)")
    ap.add_argument("--append", action="store_true", help="Append to existing labels.json instead of replacing")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load existing records if appending
    labels_path = os.path.join(OUT_DIR, "labels.json")
    if args.append and os.path.exists(labels_path):
        with open(labels_path) as f:
            records = json.load(f)
        print(f"Appending to existing {len(records)} records")
    else:
        records = []

    global_i = len(records)

    if args.videos:
        videos = [(p, _tag_from_path(p)) for p in args.videos]
    else:
        videos = DEFAULT_VIDEOS

    for vid_path, tag in videos:
        global_i = extract(vid_path, tag, args.n, args.show, OUT_DIR, records, global_i)

    if args.show:
        cv2.destroyAllWindows()

    with open(labels_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nTotal crops: {len(records)}  →  {OUT_DIR}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
