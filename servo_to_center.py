#!/usr/bin/env python3
import cv2, numpy as np, time, threading, os, re
import cv_algorithms as cva
from cv_algorithms import *

# ── Config ────────────────────────────────────────────────────────────────────
PI_HOST, PI_PORT = "192.168.0.21", 8080
XY_CAM    = 0          # bottom camera (bore -> X/Y). Was cam1 in older recordings; swap if panels look wrong
YZ_CAM    = 1          # side camera (bottom of shaft -> Z)
XY_ZOOM   = 2.75       # Pi digital zoom, set automatically. The 10.8.26 footage (1.0px bore detection) is ~2.75x
YZ_ZOOM   = 1.0        # side detection was tested at 1x
ARM_IPS   = {1: "192.168.0.3", 2: "192.168.0.4", 3: "192.168.0.5"}   # --arm N, same numbering as the GUI
ARM_ROLES = {1: "extruder", 2: "light pipe", 3: "pick-and-place"}

GAIN         = 3.0     # 1/s — XY speed = GAIN x remaining distance (from the calibrated bottom-cam Jacobian)
DEAD_BAND    = 5       # px — XY done within this radius of the bottom-cam crosshair
Z_DEAD_BAND  = 1.5     # px — Z done when the shaft bottom is within this many rows of the side-cam center line
Z_SETTLE     = 8       # frames the bottom must stay inside Z_DEAD_BAND (robot stopped) before Z counts as aligned
Z_GAIN       = 3e-5    # m/s per px of side-cam row error (100 px -> 3 mm/s). Only up/down is needed, no calibration
Z_MAX_SPEED  = 0.003   # m/s. No minimum: speed shrinks with the error, so it eases onto the line, no overshoot
SIDE_FLIPPED = False   # True if the side cam is upside down (tip goes DOWN on screen when the arm goes up)
MAX_SPEED    = 0.020   # m/s (20 mm/s) — XY servo cap and manual jog speed
ACCEL        = 0.5     # m/s²
STEP_TIME    = 0.10    # s per speedL command
CAL_DIST     = 0.0005  # m per calibration jog (0.5 mm); each snapshot is a median of CAL_FRAMES detections
CAL_FRAMES   = 5       # so the few-px side-cam shift isn't swamped by single-frame noise
TRAVEL_LIMIT = 0.010   # m — z/SPACE stop if the TCP moves further than this from where they started
# ─────────────────────────────────────────────────────────────────────────────


def z_velocity(e_side):
    """Step 1 (z key): robot Z only, from the side cam's row error. Up/down is known, so no calibration.
    e_side: (cols, rows) from the shaft bottom to the side-cam crosshair; rows > 0 means the tip is above center."""
    ev = e_side[1]
    speed = min(Z_GAIN * abs(ev), Z_MAX_SPEED)
    down = (ev > 0) != SIDE_FLIPPED
    return [0.0, 0.0, -speed if down else speed, 0.0, 0.0, 0.0]


def xy_velocity(e_xy, cal):
    """Step 3 (SPACE): robot X/Y only, from the bottom cam through the calibrated Jacobian. Z and rotation stay put."""
    v = GAIN * (cal["xy_inv"] @ np.asarray(e_xy, dtype=float))
    n = float(np.linalg.norm(v))
    if n > MAX_SPEED:
        v *= MAX_SPEED / n
    return [float(v[0]), float(v[1]), 0.0, 0.0, 0.0, 0.0]


def _would_do(e_xy, e_side):
    """Offline replay: which step would be moving on this frame."""
    if e_side is not None and abs(e_side[1]) >= Z_DEAD_BAND:
        return "Z (step 1)"
    if e_xy is None:
        return "XY: no bore detection"
    return "XY (step 3)" if np.hypot(*e_xy) >= DEAD_BAND else "CENTERED"


def _ema(prev, tip):
    """EMA on a detection; drops to None when the detection is lost so the arm never acts on a stale position."""
    if tip is None:
        return None
    t = np.array(tip, dtype=float)
    return t if prev is None else SMOOTH_ALPHA * t + (1 - SMOOTH_ALPHA) * prev


def _status(phase, e_xy, e_side, v=None):
    xy = f"{np.hypot(*e_xy):.1f}px" if e_xy is not None else "--"
    z = f"{e_side[1]:+.1f}px" if e_side is not None else "--"
    s = f"{phase}   XY err {xy}   Z err {z}"
    if v is not None:
        s += f"   v=({v[0]*1e3:.1f}, {v[1]*1e3:.1f}, {v[2]*1e3:.1f}) mm/s"
    return s


def _composite(panel_xy, panel_yz, status, notice="", height=540):
    def _scale(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * height / h), height))
    out = _scale(panel_xy) if panel_yz is None else np.hstack([_scale(panel_xy), _scale(panel_yz)])
    cv2.putText(out, status, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if notice:   # last key result (calibration outcome, why SPACE was refused, ...) so it isn't only in the terminal
        cv2.putText(out, notice, (10, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(out, notice, (10, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def calibrate(ctrl, grab_xy, last_tip):
    """Step 2 (c key): jog +X and +Y (translation only) and watch the bottom cam. The side cam isn't used: it only
    drives Z, and up/down needs no calibration.
    Returns ({"xy_inv": 2x2 m per px} or None on failure, one-line message)."""
    def fail(msg):
        print("  " + msg)
        return None, "Calibration failed: " + msg

    def snap():
        time.sleep(0.08)
        pts = []
        for _ in range(CAL_FRAMES):
            b = detect_bottom(grab_xy(), last_tip)[0]
            if b is not None: pts.append(b)
            time.sleep(0.04)                      # let the stream deliver a new frame
        return tuple(float(x) for x in np.median(pts, axis=0)) if len(pts) > CAL_FRAMES // 2 else None

    def jog(v):
        spd = 0.0025                              # 2.5 mm/s -> 0.2 s per 0.5 mm jog
        t   = CAL_DIST / spd
        ctrl.speedL([x * spd for x in v], ACCEL, t)
        time.sleep(t + 0.15)

    print(f"Calibrating XY — keep the bore in the bottom view (jogs +X, +Y by {CAL_DIST*1000:g} mm, translation only)...")
    cols = []
    for axis, name in enumerate("XY"):
        b0 = snap()
        unit = [0.0] * 6
        unit[axis] = 1.0
        jog(unit)
        b1 = snap()
        jog([-u for u in unit])
        if b0 is None or b1 is None:
            print(f"  bottom {b0}->{b1}")
            return fail(f"bottom cam lost the bore on the +{name} jog")
        d = np.subtract(b1, b0)
        cols.append(d / CAL_DIST)
        print(f"  +{name}: bottom ({d[0]:+.1f}, {d[1]:+.1f})px")

    J_xy = np.column_stack(cols)                  # px per m, 2x2 (robot X, Y)
    if abs(np.linalg.det(J_xy * CAL_DIST)) < 1 or np.linalg.cond(J_xy) > 300:
        return fail("bottom cam X/Y poorly conditioned - move needle and retry")
    msg = (f"XY calibrated: bottom cam ~{np.mean(np.linalg.svd(J_xy, compute_uv=False)) / 1000:.1f} px/mm. "
           f"Step 3: press SPACE to center XY")
    print("  " + msg)
    return {"xy_inv": np.linalg.inv(J_xy)}, msg


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
    print(f"Zoom set on the Pi: XY {XY_ZOOM}x, side {YZ_ZOOM}x (re-applied before calibrating and centering)")
    print("Step 1: z = side cam moves robot Z until the shaft bottom is on its center row (no calibration)")
    print("Step 2: c = calibrate XY (+X/+Y jogs, bottom cam only)")
    print("Step 3: SPACE = bottom cam centers X/Y. Translation only, Z untouched.")
    print("WASD=jog XY  r/f=jog Z  z/SPACE again or any jog = stop  q=quit")
    cv2.namedWindow("servo", cv2.WINDOW_NORMAL)

    JOGS = {ord('w'): [0,1,0,0,0,0], ord('s'): [0,-1,0,0,0,0],
            ord('a'): [-1,0,0,0,0,0], ord('d'): [1,0,0,0,0,0],
            ord('r'): [0,0,1,0,0,0], ord('f'): [0,0,-1,0,0,0]}

    cal, mode, p0, z_best, last_phase = None, None, None, None, None    # mode: None, "Z" (step 1) or "XY" (step 3)
    last_tip, s_xy, s_yz, bore_method, z_in = None, None, None, None, 0
    notice, view = "Click this window. Step 1: press z to align Z", None
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
            bore_method = cva.last_bore_method
            last_tip = tip or last_tip
            lk_pt    = np.float32(tip) if tip else None
            lk_n     = 0
        prev_gray = curr_gray
        s_xy = _ema(s_xy, tip)

        # ── Side cam: bottom of shaft -> Z ───────────────────────────────────
        tip_yz, side_lines = detect_side(side)
        s_yz = _ema(s_yz, tip_yz)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("z"):
            if mode == "Z":
                mode, notice = None, "Z align stopped"
            else:
                set_zoom()
                mode, p0, z_best, z_in = "Z", np.array(recv.getActualTCPPose()[:3]), None, 0
                notice = "Step 1: moving Z until the needle bottom is on the side cam's center line..."
            print(notice)
        if key == 32:
            if mode == "XY":
                mode, notice = None, "XY centering stopped"
            elif cal is None:
                notice = "SPACE ignored: XY not calibrated yet - press c first (step 2)"
            else:
                set_zoom()
                mode, p0 = "XY", np.array(recv.getActualTCPPose()[:3])
                notice = "Step 3: centering XY..."
            print(notice)
        if key == ord("c"):
            mode = None
            ctrl.speedL([0]*6, ACCEL, STEP_TIME)
            if view is not None:
                cv2.imshow("servo", _composite(*view, "Calibrating XY... (window freezes ~3 s)"))
                cv2.waitKey(1)
            set_zoom()
            new_cal, notice = calibrate(ctrl, grab_xy, last_tip)
            if new_cal is not None:
                cal = new_cal
            elif cal is not None:
                notice += " (keeping previous calibration)"
            for _ in range(100):                  # drop keys pressed while frozen, so a queued SPACE can't start
                if cv2.waitKey(1) == -1: break
        if key in JOGS:
            if mode: notice = "Jogged - stopped"
            mode = None
            ctrl.speedL([j * MAX_SPEED for j in JOGS[key]], ACCEL, STEP_TIME)
            continue

        H, W = frame.shape[:2]
        CX, CY = W // 2, H // 2
        e_xy   = (CX - s_xy[0], CY - s_xy[1]) if s_xy is not None else None
        e_side = (side.shape[1] // 2 - s_yz[0], side.shape[0] // 2 - s_yz[1]) if s_yz is not None else None

        v, phase = [0.0] * 6, "idle"
        if mode == "Z":
            if e_side is None:
                phase = "Z: side cam can't see the needle"
            else:
                ev = abs(e_side[1])
                z_best = ev if z_best is None else min(z_best, ev)
                z_in = z_in + 1 if ev < Z_DEAD_BAND else 0
                if z_in >= Z_SETTLE:              # held on the line with the arm stopped, not just passing through
                    mode, phase = None, "Z ALIGNED"
                    notice = (f"Z aligned ({e_side[1]:+.1f}px). Step 2: press c to calibrate XY "
                              f"(or SPACE if already calibrated)")
                    print(notice)
                    print("  TCP pose:", [round(float(p), 5) for p in recv.getActualTCPPose()])
                elif z_in:
                    phase = "Z settling"          # inside the band: hold still and check it stays there
                elif ev > z_best + 30:            # getting further away: the side cam's up/down is the other way
                    mode, phase = None, "Z STOPPED"
                    notice = "Z moved the wrong way - set SIDE_FLIPPED = True in servo_to_center.py"
                    print(notice)
                else:
                    phase, v = "Z", z_velocity(e_side)
        elif mode == "XY":
            if e_xy is None:
                phase = "XY: no bore detection"
            elif np.hypot(*e_xy) < DEAD_BAND:
                phase = "CENTERED"
            else:
                phase, v = "XY", xy_velocity(e_xy, cal)

        stip = (int(s_xy[0]), int(s_xy[1])) if s_xy is not None else None
        syz  = (float(s_yz[0]), float(s_yz[1])) if s_yz is not None else None
        err  = float(np.hypot(*e_xy)) if e_xy is not None else float("inf")
        panel_xy = draw_overlay(frame, stip, CX, CY, err, v[0], v[1], mode is not None, circles, lk_mode,
                                method=bore_method if tip else None)
        panel_yz = draw_side_overlay(side, syz, side_lines, z_band=Z_DEAD_BAND)
        status = _status(phase, e_xy, e_side, v)
        view = (panel_xy, panel_yz, status)
        cv2.imshow("servo", _composite(*view, notice))

        if mode:
            moved = float(np.linalg.norm(np.array(recv.getActualTCPPose()[:3]) - p0))
            if moved > TRAVEL_LIMIT:
                mode, v = None, [0.0] * 6
                notice = f"TCP moved {moved*1e3:.1f} mm since starting (limit {TRAVEL_LIMIT*1e3:.0f} mm) - stopped"
                print(notice)
        ctrl.speedL(v, ACCEL, STEP_TIME)

        if mode and phase != last_phase:
            print(_status(phase, e_xy, e_side, v), flush=True)
            if phase == "CENTERED":
                print("  TCP pose at center:", [round(float(p), 5) for p in recv.getActualTCPPose()])
        last_phase = phase if mode else None

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

    while True:
        if not paused:
            ok_xy, frame_xy = cap_xy.read()
            if not ok_xy:
                # Loop both videos
                cap_xy.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if cap_yz:
                    cap_yz.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_gray = None; lk_pt = None; lk_n = 0
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
                                circles, lk_mode, bore_circle, det_mode, cva.last_bore_method if tip else None)

        # ── YZ side detection ────────────────────────────────────────────────
        tip_yz = None
        if cap_yz:
            tip_yz, side_lines = detect_side(frame_yz)
            panel_yz = draw_side_overlay(frame_yz, tip_yz, side_lines, det_mode, z_band=Z_DEAD_BAND)
        else:
            panel_yz = None

        e_xy   = (CX - tip[0], CY - tip[1]) if tip else None
        e_side = (frame_yz.shape[1] // 2 - tip_yz[0], frame_yz.shape[0] // 2 - tip_yz[1]) if tip_yz else None
        cv2.imshow("test", _composite(panel_xy, panel_yz, "would do: " + _status(_would_do(e_xy, e_side), e_xy, e_side)))
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
            prev_gray = None; lk_pt = None; lk_n = 0

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
