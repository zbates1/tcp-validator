"""
Extract raw frames from a video into a structured dataset folder.

Usage:
    python extract_frames.py video.mp4
    python extract_frames.py video.mp4 --name my_dataset
    python extract_frames.py video.mp4 --step 3 --out datasets/
    python extract_frames.py video.mp4 --n 300      # ~300 frames evenly sampled
"""
import argparse, cv2, os, re


def extract_frames(video_path, out_dir, step=1, prefix=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(video_path)}: {total} frames  step={step}  → {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(os.path.join(out_dir, f"{prefix}frame_{i:05d}.jpg"), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--name", default=None,
                    help="Dataset folder name (default: derived from filename)")
    ap.add_argument("--step", type=int, default=None,
                    help="Extract every Nth frame (default: auto from --n)")
    ap.add_argument("--n", type=int, default=None,
                    help="Target number of frames to extract (overrides --step)")
    ap.add_argument("--out", default="datasets",
                    help="Parent directory for datasets/ (default: datasets/)")
    ap.add_argument("--prefix", default="",
                    help="Filename prefix, so several videos can share one dataset folder")
    args = ap.parse_args()

    # Derive dataset name from filename if not given
    if args.name is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        # Use last component after last underscore-group, or full base if short
        args.name = re.sub(r'[^\w]', '_', base)[-40:]

    cap_tmp = cv2.VideoCapture(args.video)
    total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()

    if args.step is not None:
        step = args.step
    elif args.n is not None:
        step = max(1, total // args.n)
    else:
        step = 1

    base = os.path.dirname(os.path.abspath(args.video))
    datasets_dir = os.path.join(base, args.out) if not os.path.isabs(args.out) else args.out
    unlabeled_dir = os.path.join(datasets_dir, args.name, "unlabeled")
    labeled_dir   = os.path.join(datasets_dir, args.name, "labeled")
    os.makedirs(labeled_dir, exist_ok=True)

    extract_frames(args.video, unlabeled_dir, step, args.prefix)
    print(f"\nDataset ready at: {os.path.join(datasets_dir, args.name)}")
    print(f"  Label with: python labeler_ui.py {os.path.join(datasets_dir, args.name)}")


if __name__ == "__main__":
    main()
