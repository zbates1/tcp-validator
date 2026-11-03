#!/usr/bin/env python3
import cv2, numpy as np, time, threading, os, re
from cv_algorithms import *

# ── Config ────────────────────────────────────────────────────────────────────
PI_HOST, PI_PORT = "192.168.0.21", 8080
XY_CAM    = 0          # bottom camera (bore -> X/Y). Was cam1 in older recordings; swap if panels look wrong
YZ_CAM    = 1          # side camera (bottom of shaft -> Z, plus a coarse horizontal pre-center)
XY_ZOOM   = 2.75       # Pi digital zoom, set automatically. The 10.8.26 footage (1.0px bore detection) is ~2.75x
YZ_ZOOM   = 1.0        # side detection was tested at 1x
ARM_IPS   = {1: "192.168.0.3", 2: "192.168.0.4", 3: "192.168.0.5"}   # --arm N, same numbering as the GUI
ARM_ROLES = {1: "extruder", 2: "light pipe", 3: "pick-and-place"}

GAIN         = 3.0     # 1/s — commanded speed = GAIN x remaining distance (from the calibrated Jacobians)
DEAD_BAND    = 5       # px — XY done within this radius of the bottom-cam crosshair
Z_DEAD_BAND  = 5       # px — side-cam rows; Z is re-checked after XY and re-entered past 2x this
SIDE_H_TOL   = 50      # px — side-cam columns; only a coarse pre-center, the bottom cam owns X/Y
MAX_SPEED    = 0.020   # m/s (20 mm/s)
ACCEL        = 0.5     # m/s²
STEP_TIME    = 0.10    # s per speedL command
CAL_DIST     = 0.003   # m per calibration jog (3 mm) — needs to be large enough for reliable px measurement
TRAVEL_LIMIT = 0.010   # m — servo disarms if the TCP moves further than this from where it was armed
# ─────────────────────────────────────────────────────────────────────────────


def servo_phase(e_xy, e_side, side_active, h_done):
    """Decide which camera drives this frame.
    SIDE: shaft bottom onto the side cam's center row (tight, robot Z) and near its center column (coarse, first
          pass only). XY: bottom cam centers the bore. Z is re-checked after XY so both end inside their dead bands.
    e_xy / e_side: (cols, rows) from tip to that camera's crosshair, or None if not detected.
    Returns (phase, side_active, h_done)."""
    if e_side is not None:
        eu, ev = e_side
        h_done = h_done or abs(eu) < SIDE_H_TOL
        if side_active and h_done and abs(ev) < Z_DEAD_BAND:
            side_active = False
        elif not side_active and abs(ev) > 2 * Z_DEAD_BAND:
            side_active = True
    if side_active:
        return ("SIDE" if e_side is not None else "SIDE: no side detection"), side_active, h_done
    if e_xy is None:
        return "XY: no bore detection", side_active, h_done
    if np.hypot(*e_xy) >= DEAD_BAND:
        return "XY", side_active, h_done
    if e_side is None:
        return "XY done: no side detection", side_active, h_done
    if abs(e_side[1]) < Z_DEAD_BAND:
        return "CENTERED", side_active, h_done
    return "SIDE", True, h_done      # XY done but Z drifted inside the hysteresis band: touch Z up


def servo_velocity(phase, e_xy, e_side, h_done, cal):
    """speedL vector for the phase. Translation only — the needle is never rotated."""
    if phase == "SIDE":
        eu, ev = e_side
        if h_done:
            d = np.array([0.0, 0.0, ev / cal["kz"]])                 # pure robot Z
        else:
            d = cal["side_pinv"] @ np.array([eu, ev], dtype=float)   # smallest XYZ move in the side cam's view plane
    elif phase == "XY":
        d = np.append(cal["xy_inv"] @ np.asarray(e_xy, dtype=float), 0.0)
    else:
        return [0.0] * 6
    v = GAIN * d
    n = float(np.linalg.norm(v))
    if n > MAX_SPEED:
        v *= MAX_SPEED / n
    return [float(v[0]), float(v[1]), float(v[2]), 0.0, 0.0, 0.0]


def _ema(prev, tip):
    """EMA on a detection; drops to None when the detection is lost so the arm never acts on a stale position."""
    if tip is None:
        return None
    t = np.array(tip, dtype=float)
    return t if prev is None else SMOOTH_ALPHA * t + (1 - SMOOTH_ALPHA) * prev


def _status(phase, e_xy, e_side, v=None):
    xy = f"{np.hypot(*e_xy):.1f}px" if e_xy is not None else "--"
    sd = f"({e_side[0]:+.0f}, {e_side[1]:+.0f})px" if e_side is not None else "--"
    s = f"{phase}   XY err {xy}   side err (col,row) {sd}"
    if v is not None:
        s += f"   v=({v[0]*1e3:.1f}, {v[1]*1e3:.1f}, {v[2]*1e3:.1f}) mm/s"
    return s


def _composite(panel_xy, panel_yz, status, height=540):
    def _scale(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * height / h), height))
    out = _scale(panel_xy) if panel_yz is None else np.hstack([_scale(panel_xy), _scale(panel_yz)])
    cv2.putText(out, status, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def calibrate(ctrl, grab_xy, grab_yz, last_tip):
    """Jog +X, +Y, +Z (translation only) and watch both cameras.
    Returns {"xy_inv": 2x2, "side_pinv": 3x2, "kz": side rows per m of Z}, or None on failure."""
    def snap():
        time.sleep(0.08)
        return detect_bottom(grab_xy(), last_tip)[0], detect_side(grab_yz())[0]

    def jog(v):
        spd = 0.005
        t   = CAL_DIST / spd
        ctrl.speedL([x * spd for x in v], ACCEL, t)
        time.sleep(t + 0.15)

    print(f"Calibrating — keep the tip in both views (jogs +X, +Y, +Z by {CAL_DIST*1000:.0f} mm, translation only)...")
    cols_xy, cols_side = [], []
    for axis, name in enumerate("XYZ"):
        b0, s0 = snap()
        unit = [0.0] * 6
        unit[axis] = 1.0
        jog(unit)
        b1, s1 = snap()
        jog([-u for u in unit])
        need_bottom = axis < 2                    # the Z jog only needs the side cam
        if s0 is None or s1 is None or (need_bottom and (b0 is None or b1 is None)):
            print(f"  Lost the tip on the +{name} jog (bottom {b0}->{b1}, side {s0}->{s1})"); return None
        if need_bottom:
            cols_xy.append(np.subtract(b1, b0) / CAL_DIST)
        cols_side.append(np.subtract(s1, s0) / CAL_DIST)
        bottom_px = tuple(int(x) for x in np.subtract(b1, b0)) if need_bottom else "-"
        print(f"  +{name}: bottom {bottom_px}px  side {tuple(int(x) for x in np.subtract(s1, s0))}px")

    J_xy, J_side = np.column_stack(cols_xy), np.column_stack(cols_side)   # px per m: 2x2 (X,Y), 2x3 (X,Y,Z)
    if abs(np.linalg.det(J_xy * CAL_DIST)) < 1 or np.linalg.cond(J_xy) > 300:
        print("  Bottom cam X/Y poorly conditioned — move needle and retry"); return None
    kz = float(J_side[1, 2])
    if abs(kz * CAL_DIST) < 3:
        print("  Side cam barely saw the Z jog — check the tip is in view and retry"); return None
    if np.linalg.svd(J_side * CAL_DIST, compute_uv=False)[1] < 3:
        print("  Side cam barely saw the X/Y jogs — can't pre-center horizontally"); return None
    print(f"  bottom cam ~{np.mean(np.linalg.svd(J_xy, compute_uv=False)) / 1000:.1f} px/mm, "
          f"side cam Z {abs(kz) / 1000:.1f} px/mm — press SPACE to servo")
    return {"xy_inv": np.linalg.inv(J_xy), "side_pinv": np.linalg.pinv(J_side), "kz": kz}


def set_zoom():
    """Set both cameras' zoom on the Pi, so the servo always runs at the zoom it was calibrated at."""
    import urllib.request
    for idx, level in ((XY_CAM, XY_ZOOM), (YZ_CAM, YZ_ZOOM)):
        try:
            urllib.request.urlopen(f"http://{PI_HOST}:{PI_PORT}/zoom?cam={idx}&level={level}", timeout=2).read()
        except OSError as e:
            print(f"WARNING: couldn't set cam{idx} zoom to {level}x ({e})")
    time.sleep(0.3)   # let frames at the new zoom arrive


def open_stream(idx, stop):
    """Background reader for one Pi MJPEG stream; returns (cap, grab) where grab() copies the newest frame."""
    url = f"http://{PI_HOST}:{PI_PORT}/cam{idx}.mjpg"
    cap = cv2.VideoCapture(url)
    assert cap.isOpened(), f"Cannot open {url}"
    lock, buf = threading.Lock(), [None]
    def _read():
        while not stop.is_set():
            ok, f = cap.read()
            if ok:
                with lock: buf[0] = f
    threading.Thread(target=_read, daemon=True).start()
    while buf[0] is None: time.sleep(0.01)
    def grab():
        with lock: return buf[0].copy()
    return cap, grab


def main(arm=1):
    import rtde_control, rtde_receive

    ip = ARM_IPS[arm]
    try:
        ctrl = rtde_control.RTDEControlInterface(ip)
        recv = rtde_receive.RTDEReceiveInterface(ip)
    except Exception as e:
        raise SystemExit(f"Can't connect to arm {arm} ({ARM_ROLES[arm]}) at {ip}: {e}\n"
                         f"Check it's powered on, reachable (ping {ip}) and in Remote Control mode, "
                         f"or pick another arm with --arm.")
    print(f"Connected to arm {arm} ({ARM_ROLES[arm]}) at {ip}")

    stop = threading.Event()
    set_zoom()
    cap_xy, grab_xy = open_stream(XY_CAM, stop)
    cap_yz, grab_yz = open_stream(YZ_CAM, stop)
    if max(crosshair_score(grab_xy()), crosshair_score(grab_yz())) > 30:   # clean frames score 0, burned-in 73+
        print("WARNING: the Pi is drawing its crosshair into the stream. Copy the current pi_stream_server.py to the\n"
              "         Pi and restart it. Erasing it here works, but side detection fails when the needle sits on\n"
              "         the center line.")
        raw_xy, raw_yz = grab_xy, grab_yz
        grab_xy = lambda: strip_crosshair(raw_xy())
        grab_yz = lambda: strip_crosshair(raw_yz())
    print(f"XY = cam{XY_CAM} (left panel), YZ = cam{YZ_CAM} (right panel) — swap XY_CAM/YZ_CAM if they look wrong")
    print(f"Zoom set on the Pi: XY {XY_ZOOM}x, side {YZ_ZOOM}x (re-applied before calibrating and when arming)")
    print("Side cam first (Z exact, horizontal coarse), then the bottom cam centers X/Y. Translation only.")
    print("WASD=jog XY  r/f=jog Z  c=calibrate (required before servo)  SPACE=servo  q=quit")
    cv2.namedWindow("servo", cv2.WINDOW_NORMAL)

    JOGS = {ord('w'): [0,1,0,0,0,0], ord('s'): [0,-1,0,0,0,0],
            ord('a'): [-1,0,0,0,0,0], ord('d'): [1,0,0,0,0,0],
            ord('r'): [0,0,1,0,0,0], ord('f'): [0,0,-1,0,0,0]}

    cal, armed, side_active, h_done, p0, last_phase = None, False, True, False, None, None
    last_tip, s_xy, s_yz = None, None, None
    prev_gray, lk_pt, lk_n = None, None, 0

    while True:
        frame, side = grab_xy(), grab_yz()

        # ── Bottom cam: bore -> X/Y ──────────────────────────────────────────
        circles   = []
        lk_mode   = False
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tip = None
        if lk_pt is not None and prev_gray is not None and lk_n < LK_REACQUIRE_EVERY:
            new_pt = track_bore_lk(prev_gray, curr_gray, lk_pt)
            if new_pt is not None:
                lk_pt    = new_pt
                tip      = (int(round(float(new_pt[0]))), int(round(float(new_pt[1]))))
                last_tip = tip
                lk_mode  = True
                lk_n    += 1
        if not lk_mode:
            tip, circles, _ = detect_bottom(frame, last_tip)
            last_tip = tip or last_tip
            lk_pt    = np.float32(tip) if tip else None
            lk_n     = 0
        prev_gray = curr_gray
        s_xy = _ema(s_xy, tip)

        # ── Side cam: bottom of shaft -> Z (+ coarse horizontal) ─────────────
        tip_yz, side_lines = detect_side(side)
        s_yz = _ema(s_yz, tip_yz)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == 32:
            if cal is None:
                print("Press c to calibrate first")
            else:
                armed = not armed
                if armed:
                    set_zoom()
                    side_active, h_done, p0 = True, False, np.array(recv.getActualTCPPose()[:3])
                print("SERVO", "ON" if armed else "OFF")
        if key == ord("c"):
            armed = False
            ctrl.speedL([0]*6, ACCEL, STEP_TIME)
            set_zoom()
            new_cal = calibrate(ctrl, grab_xy, grab_yz, last_tip)
            if new_cal is not None:
                cal = new_cal
        if key in JOGS:
            armed = False
            ctrl.speedL([j * MAX_SPEED for j in JOGS[key]], ACCEL, STEP_TIME)
            continue

        H, W = frame.shape[:2]
        CX, CY = W // 2, H // 2
        e_xy   = (CX - s_xy[0], CY - s_xy[1]) if s_xy is not None else None
        e_side = (side.shape[1] // 2 - s_yz[0], side.shape[0] // 2 - s_yz[1]) if s_yz is not None else None
        phase, side_active, h_done = servo_phase(e_xy, e_side, side_active, h_done)
        v = servo_velocity(phase, e_xy, e_side, h_done, cal) if cal else [0.0] * 6

        stip = (int(s_xy[0]), int(s_xy[1])) if s_xy is not None else None
        syz  = (int(s_yz[0]), int(s_yz[1])) if s_yz is not None else None
        err  = float(np.hypot(*e_xy)) if e_xy is not None else float("inf")
        panel_xy = draw_overlay(frame, stip, CX, CY, err, v[0], v[1], armed, circles, lk_mode)
        panel_yz = draw_side_overlay(side, syz, side_lines)
        status = _status(phase, e_xy, e_side, v) + ("" if cal else "   (press c to calibrate)")
        cv2.imshow("servo", _composite(panel_xy, panel_yz, status))

        if armed:
            moved = float(np.linalg.norm(np.array(recv.getActualTCPPose()[:3]) - p0))
            if moved > TRAVEL_LIMIT:
                armed = False
                print(f"TCP moved {moved*1e3:.1f} mm since arming (limit {TRAVEL_LIMIT*1e3:.0f} mm) — SERVO OFF")
        ctrl.speedL(v if armed else [0]*6, ACCEL, STEP_TIME)

        if armed and phase != last_phase:
            print(_status(phase, e_xy, e_side, v), flush=True)
            if phase == "CENTERED":
                print("  TCP pose at center:", [round(float(p), 5) for p in recv.getActualTCPPose()])
        last_phase = phase if armed else None

    stop.set()
    ctrl.speedL([0]*6, ACCEL)
    cap_xy.release(); cap_yz.release(); cv2.destroyAllWindows()
    print("Stopped.")


def test_video(xy_path, yz_path=None):
    """Offline detection test — no robot needed.
    Shows XY bore detection (left) and YZ side-view (right) side by side, plus which phase the servo would be in.
    Keys: q=quit  space=pause  r=restart  m=toggle center/edge view"""
    cap_xy = cv2.VideoCapture(xy_path)
    assert cap_xy.isOpened(), f"Cannot open XY video: {xy_path}"
    cap_yz = cv2.VideoCapture(yz_path) if yz_path else None
    if cap_yz and not cap_yz.isOpened():
        print(f"Warning: cannot open YZ video {yz_path} — showing XY only")
        cap_yz = None

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)

    last_tip    = None
    paused      = False
    prev_gray   = None
    lk_pt       = None
    lk_n        = 0
    det_mode    = 0    # 0 = center/tip  1 = radius/edge  (toggle with 'm')
    side_active = True
    h_done      = False

    while True:
        if not paused:
            ok_xy, frame_xy = cap_xy.read()
            if not ok_xy:
                # Loop both videos
                cap_xy.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if cap_yz:
                    cap_yz.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_gray = None; lk_pt = None; lk_n = 0; side_active, h_done = True, False
                continue
            if cap_yz:
                ok_yz, frame_yz = cap_yz.read()
                if not ok_yz:
                    cap_yz.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, frame_yz = cap_yz.read()

        # ── XY bore detection ────────────────────────────────────────────────
        curr_gray = cv2.cvtColor(frame_xy, cv2.COLOR_BGR2GRAY)
        circles   = []
        lk_mode   = False

        if lk_pt is not None and prev_gray is not None and lk_n < LK_REACQUIRE_EVERY:
            new_pt = track_bore_lk(prev_gray, curr_gray, lk_pt)
            if new_pt is not None:
                lk_pt    = new_pt
                tip      = (int(round(float(new_pt[0]))), int(round(float(new_pt[1]))))
                last_tip = tip
                lk_mode  = True
                lk_n    += 1

        bore_circle = None
        if not lk_mode:
            tip, circles, bore_circle = detect_bottom(frame_xy, last_tip)
            last_tip = tip or last_tip
            lk_pt    = np.float32(tip) if tip else None
            lk_n     = 0

        if not paused:
            prev_gray = curr_gray

        H, W = frame_xy.shape[:2]
        CX, CY = W // 2, H // 2
        err = ((CX - tip[0])**2 + (CY - tip[1])**2)**0.5 if tip else float("inf")
        panel_xy = draw_overlay(frame_xy, tip, CX, CY, err, 0, 0, False,
                                circles, lk_mode, bore_circle, det_mode)

        # ── YZ side detection ────────────────────────────────────────────────
        tip_yz = None
        if cap_yz:
            tip_yz, side_lines = detect_side(frame_yz)
            panel_yz = draw_side_overlay(frame_yz, tip_yz, side_lines, det_mode)
        else:
            panel_yz = None

        e_xy   = (CX - tip[0], CY - tip[1]) if tip else None
        e_side = (frame_yz.shape[1] // 2 - tip_yz[0], frame_yz.shape[0] // 2 - tip_yz[1]) if tip_yz else None
        phase, side_active, h_done = servo_phase(e_xy, e_side, side_active, h_done)
        cv2.imshow("test", _composite(panel_xy, panel_yz, "would do: " + _status(phase, e_xy, e_side)))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"): break
        if key == 32: paused = not paused
        if key == ord("m"):
            det_mode = 1 - det_mode
            label = "RADIUS / EDGE" if det_mode == 1 else "CENTER / TIP"
            print(f"Mode → {label}")
        if key == ord("r"):
            cap_xy.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cap_yz: cap_yz.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prev_gray = None; lk_pt = None; lk_n = 0; side_active, h_done = True, False

    cap_xy.release()
    if cap_yz: cap_yz.release()
    cv2.destroyAllWindows()


def _is_xy_video(path):
    """The bottom cam is the zoomed one in every recording so far (cam1_z3.75x, cam0_z2.50x, *_xy_cam)."""
    name = os.path.basename(path).lower()
    return re.search(r"_z\d", name) is not None or "xy" in name


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Center the needle tip with both cameras, or replay videos offline.")
    ap.add_argument("videos", nargs="*", help="offline test (no robot): XY video and optional YZ video")
    ap.add_argument("--arm", type=int, choices=sorted(ARM_IPS), default=1,
                    help="arm to drive: " + ", ".join(f"{n} = {ARM_IPS[n]} ({ARM_ROLES[n]})" for n in ARM_IPS))
    args = ap.parse_args()
    if not args.videos:
        main(args.arm)
    elif len(args.videos) == 1:
        test_video(args.videos[0])
    else:
        # cam0/cam1 numbering has flipped between sessions, so pick XY by its zoom tag (or "xy" in the name)
        xy_path, yz_path = sorted(args.videos[:2], key=lambda p: not _is_xy_video(p))
        print(f"XY (bore):  {os.path.basename(xy_path)}")
        print(f"YZ (side):  {os.path.basename(yz_path)}")
        test_video(xy_path, yz_path)
