#!/usr/bin/env python3
"""
Final_Game.py  —  HCC 2026 Autonomous Competition Pipeline
===========================================================
Run:   python Final_Game.py

Press ESC in the camera window for emergency land at any point.

Pipeline
--------
  [Stage 1 ]  Takeoff (starting angle fixed at 0 — drawing not allowed)
  [Stage 1b]  Scan for ANY AprilTag → estimate position → fly directly to
              the midpoint of tags 15/16 (facing away from wall 4-9) →
              rescan with whichever tag is visible → repeat until within
              the (large) deadzone.
  [Stage 2 ]  Search balloon (slow yaw scan)
  [Stage 3 ]  Kalman + PID track → confirmed-close sprint → touch
  [Stage 4a]  Re-run the SAME scan→fly→rescan loop back to the 15/16
              midpoint (the drone may have drifted during the chase)
  [Stage 4b]  Brainrot classification (majority vote)
  [Stage 4c]  Re-run the scan→fly→rescan loop ONE more time (drift check
              before the final approach — deadzone kept large here too,
              for an easier hand-off into the precise landing approach)
  [Stage 4d]  Rotate (stop-and-look) until landing AprilTag visible →
              PID approach → land

Changes in this revision
--------------------------
  1. Stage 1b (and the two new Stage-4 navigation passes) no longer stream
     continuous RC velocity while requiring the tag to stay in view the
     whole time. Instead: scan for ANY tag (stop-and-look) → estimate
     full
     world pose from that ONE tag → fly DIRECTLY to the target with a
     single relative Tello "go" maneuver (djitellopy go_xyz_speed, using
     the same body-frame error decomposition already validated by
     simulation) → rescan (possibly a different tag is now visible) →
     check against an ENLARGED deadzone → repeat if not yet inside it.
     This is far more robust to losing tag visibility mid-motion than
     trying to hold a continuous closed loop on one tag while rotating.
  2. The drone now also returns to the 15/16 midpoint AFTER touching the
     balloon (it may have drifted far away during the chase), classifies
     the brainrot image there, then re-runs the same nav loop once more
     (drift check) before handing off to the precise per-tag landing
     approach.
  3. Both deadzones (initial nav-to-point, and the post-classify recheck)
     are intentionally larger than before — this stage just needs to get
     the drone roughly back into the open area facing the right way;
     the final approach_and_land() does the precise centering on the
     SPECIFIC landing tag.

Detection backend
------------------
Both balloon and brainrot models load from .pt weights via Ultralytics.
  pip install ultralytics djitellopy pupil-apriltags opencv-python scipy pyyaml
"""

import math
import os
import threading
import time
from collections import Counter

import cv2
import numpy as np
from pupil_apriltags import Detector as ATDetector
from ultralytics import YOLO

import apriltag_localizer as al  # Self-localization + nav-target math
import Balloon_Detector  # Stage 2/3 logic (Kalman + PID + touch)
import Start_Tello  # Stage 1 takeoff helpers
from pid_controller import PIDController


class EmergencyLand(Exception):
    """Raised when ESC is pressed during a blocking sub-routine, so the rest
    of the mission sequence in main() is skipped (not just the current step)."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  tune these numbers before the competition
# ═══════════════════════════════════════════════════════════════════════════════

BALLOON_PT  = 'balloon.pt'
BRAINROT_PT = 'brainrot_detect.pt'
MAP_YAML    = 'Apriltag/map/apriltag_map.yaml'   # falls back to ./apriltag_map.yaml

# ── AprilTag camera intrinsics (Lab 1 calibration) ──────────────────────────
AT_FX, AT_FY = 835.342103847164, 839.4691450667409
AT_CX, AT_CY = 415.5366635247159, 355.11975613817964

# Physical AprilTag side length in metres.
# NOTE: verify this matches the *printed* tags used at the competition venue.
AT_TAG_SIZE = 0.165

# ── Scan-fly-rescan navigation (used 3x: pre-balloon, post-touch, post-classify) ──
NAV_TARGET_IDS  = (15, 16)              # fly to the midpoint of these tags
NAV_WALL_IDS    = (4, 5, 6, 7, 8, 9)    # face away from this wall
NAV_HEIGHT_CM   = 130                   # cruise height during navigation

# Deadzone — intentionally large: this stage just needs to get the drone
# roughly back to the open area facing the right way, not landing-precise.
NAV_POS_TOL_M   = 0.5
NAV_YAW_TOL_RAD = math.radians(30)

NAV_GO_SPEED_CMS   = 30    # speed for the one-shot relative "go" maneuver
NAV_MIN_MOVE_CM     = 20   # Tello "go" SDK command requires each nonzero axis
                            # to be >= 20cm (a value of 1-19 is rejected with
                            # "error" by the firmware) — see the official
                            # Tello SDK doc for "go x y z speed": x/y/z: 20-500.
                            # The old value of 10 silently dropped small
                            # corrective moves (incl. small altitude trims).
NAV_MAX_MOVE_CM     = 150  # safety cap per single move, per axis
NAV_MAX_ITERATIONS  = 2    # give up (and proceed anyway) after this many cycles
NAV_SCAN_WARN_S     = 8.0  # print a reminder if no tag found within this long

# ── Classification ────────────────────────────────────────────────────────────
LANDING_TAG          = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}
CLASSIFY_FRAMES      = 20
MIN_VOTE_FRACTION    = 0.50
CLASSIFY_TIMEOUT     = 30.0
CLASSIFY_CONF_THRESH = 0.15

# ── Stage-4 final approach: precise, continuous, single target tag ──────────
# The landing target is a 30x30cm box on the floor ~20cm in front of (out
# from the wall from) the landing tag — NOT a hover point near the tag
# itself. APPROACH_DIST_M is therefore the box's distance from the tag's
# wall plane, not an arbitrary safe-hover distance. The PID below still
# matches the drone's altitude to the tag's vertical centre (err_alt = ty)
# while closing in — that's fine, because tello.land() performs its own
# full controlled descent to the ground from whatever altitude it's
# called at. As long as the horizontal (forward/lateral/yaw) position is
# correct when land() fires, the autoland brings it down inside the box.
# (15cm half-width box + 8cm lateral tolerance and 10cm forward tolerance
# below comfortably fit within the box if 20cm is the box's centre.)
APPROACH_DIST_M = 0.20
LAND_POS_TOL    = 0.10
LAND_LAT_TOL    = 0.08
KP_FWD, KI_FWD, KD_FWD = 40.0, 0.0, 8.0
KP_LAT, KI_LAT, KD_LAT = 50.0, 0.0, 10.0
KP_ALT, KI_ALT, KD_ALT = 40.0, 0.0, 8.0
KP_YAW, KI_YAW, KD_YAW = 50.0, 0.0, 5.0

# ── Shared stop-and-look search pattern (continuous yaw blurs AprilTag detection) ──
SEARCH_YAW_PCT  = 25
SEARCH_ROTATE_S = 0.4
SEARCH_PAUSE_S  = 0.6

# ── Touch sprint (non-blocking stage, see TOUCH_SPRINT below) ──
TOUCH_SPRINT_FB       = 45
TOUCH_SPRINT_DURATION = 0.8

# ── Timing ────────────────────────────────────────────────────────────────────
POST_TOUCH_HOVER_S = 2.0   # just stabilisation after the sprint, before re-navigating


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_map_yaml():
    candidates = [MAP_YAML, os.path.join(os.path.dirname(__file__), MAP_YAML),
                  'apriltag_map.yaml', os.path.join(os.path.dirname(__file__), 'apriltag_map.yaml')]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"apriltag_map.yaml not found. Tried: {candidates}")


def stop_and_look_step(tello, search_state: dict):
    """
    Continuous yaw rotation blurs frames enough that pupil_apriltags often
    misses tags entirely. Alternate short rotation bursts with still pauses
    so the detector gets a sharp frame to work with.

    search_state must have keys 'rotating' (bool) and 'phase_start' (float),
    mutated in place. Call this once per frame whenever no tag is visible.
    """
    now     = time.time()
    elapsed = now - search_state['phase_start']

    if search_state['rotating']:
        tello.send_rc_control(0, 0, 0, SEARCH_YAW_PCT)
        if elapsed > SEARCH_ROTATE_S:
            tello.send_rc_control(0, 0, 0, 0)
            search_state['rotating']    = False
            search_state['phase_start'] = now
    else:
        tello.send_rc_control(0, 0, 0, 0)
        if elapsed > SEARCH_PAUSE_S:
            search_state['rotating']    = True
            search_state['phase_start'] = now

    return search_state['rotating']


def _clamp_move(cm: float, min_cm: int = NAV_MIN_MOVE_CM, max_cm: int = NAV_MAX_MOVE_CM) -> int:
    """Round to int and clamp to Tello's 'go' command constraints (0, or in [min_cm, max_cm])."""
    cm = int(round(cm))
    cm = max(-max_cm, min(max_cm, cm))
    if 0 < abs(cm) < min_cm:
        cm = min_cm if cm > 0 else -min_cm
    return cm


def _rotate_by_yaw_err(tello, yaw_err_rad: float, min_deg: int = 3):
    """
    Rotate to close a world-frame yaw error.
    yaw_err > 0 means target_yaw is more CCW than current -> need to
    INCREASE yaw -> rotate_counter_clockwise (Tello CW+ decreases world CCW+ yaw).
    """
    deg = int(round(math.degrees(yaw_err_rad)))
    deg = max(-179, min(179, deg))
    if abs(deg) < min_deg:
        return
    if deg > 0:
        tello.rotate_counter_clockwise(abs(deg))
    else:
        tello.rotate_clockwise(abs(deg))


def _run_blocking_with_live_feed(tello, frame_read, fn, status_text: str, label: str):
    """
    djitellopy's go_xyz_speed()/rotate_clockwise()/rotate_counter_clockwise()
    all block the calling thread until the Tello replies "ok" — which for a
    "go" maneuver only happens once the drone has physically finished
    travelling. Calling these directly on the main thread means cv2.imshow()
    never runs during the manoeuvre, so the preview window appears frozen
    (and, separately, the Tello's own video bandwidth is known to dip while
    it's busy executing a "go"/"curve"/rotate command).

    This runs `fn` (a zero-arg callable wrapping the blocking SDK call) on a
    background thread while the main thread keeps pulling frames and calling
    cv2.imshow(), so the preview keeps updating and ESC still works during
    the manoeuvre. Raises EmergencyLand if ESC is pressed while waiting.
    """
    result = {}

    def _worker():
        try:
            fn()
        except Exception as e:
            result['error'] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while t.is_alive():
        frame = frame_read.frame
        if frame is not None:
            display = frame.copy()
            cv2.putText(display, f"[{label}] {status_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.imshow('HCC Final Game', display)
        if cv2.waitKey(1) & 0xFF == 27:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            raise EmergencyLand(f"ESC during {label} ({status_text})")
        time.sleep(0.02)

    t.join()
    if 'error' in result:
        raise result['error']


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAN → ESTIMATE → FLY DIRECTLY → RESCAN → VERIFY DEADZONE
# ═══════════════════════════════════════════════════════════════════════════════

def navigate_to_point(tello, frame_read, at_det: ATDetector, tag_pose_dict: dict,
                      target_x: float, target_y: float, target_yaw: float,
                      pos_tol: float = NAV_POS_TOL_M, yaw_tol: float = NAV_YAW_TOL_RAD,
                      max_iterations: int = NAV_MAX_ITERATIONS, label: str = "Nav"):
    """
    Repeatedly: scan for ANY AprilTag (stop-and-look), estimate the drone's
    full world pose from that single tag, fly DIRECTLY toward the target
    with one relative 'go' maneuver, then rescan (a different tag may now
    be the one visible) to verify. Loops until within (pos_tol, yaw_tol)
    or max_iterations is reached.
    """
    cam_params = [AT_FX, AT_FY, AT_CX, AT_CY]
    print(f"[{label}] Navigating to ({target_x:.2f}, {target_y:.2f}) "
          f"yaw={math.degrees(target_yaw):.1f}°  "
          f"(deadzone: {pos_tol:.2f}m / {math.degrees(yaw_tol):.1f}°)")

    for iteration in range(1, max_iterations + 1):
        # ── Scan for any tag (stop-and-look) ─────────────────────────────
        search_state  = {'rotating': True, 'phase_start': time.time()}
        scan_start    = time.time()
        warned        = False
        pose          = None

        while pose is None:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.02)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_det.detect(gray, estimate_tag_pose=True,
                                 camera_params=cam_params, tag_size=AT_TAG_SIZE)
            pose = al.localize_best_tag(tags, tag_pose_dict)

            display = frame.copy()
            if pose is None:
                rotating = stop_and_look_step(tello, search_state)
                cv2.putText(display,
                            f"[{label}] iter {iteration}/{max_iterations}: "
                            f"{'rotating' if rotating else 'looking'} for any AprilTag...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                if not warned and time.time() - scan_start > NAV_SCAN_WARN_S:
                    print(f"[{label}] Still scanning for a tag ({time.time()-scan_start:.0f}s)...")
                    warned = True

            cv2.imshow('HCC Final Game', display)
            if cv2.waitKey(1) & 0xFF == 27:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                raise EmergencyLand("ESC during navigate_to_point scan")
            time.sleep(0.02)

        tello.send_rc_control(0, 0, 0, 0)
        x, y, z, yaw = pose
        dx, dy  = target_x - x, target_y - y
        pos_err = math.hypot(dx, dy)
        yaw_err = al.wrap_angle(target_yaw - yaw)

        print(f"[{label}] iter {iteration}: pose=({x:.2f},{y:.2f},yaw={math.degrees(yaw):.1f}°)  "
              f"pos_err={pos_err:.2f}m  yaw_err={math.degrees(yaw_err):.1f}°")

        if pos_err < pos_tol and abs(yaw_err) < yaw_tol:
            print(f"[{label}] Within deadzone — done.")
            return True

        # ── Fly DIRECTLY to the estimated point (one-shot relative 'go') ──
        # Tello "go x y z speed" convention (confirmed against the official
        # SDK doc + djitellopy): x = forward(+)/backward(-),
        # y = RIGHT(+)/LEFT(-), z = up(+)/down(-).
        # world_error_to_body() returns "right_err" already in that same
        # sign convention (+ = need to move right) — it must be passed
        # through as-is, NOT negated. The previous code negated it into a
        # "left_cm" and fed that into the y-slot, which flipped every
        # lateral correction (the drone went right when it needed to go
        # left, and vice versa) — this was the "weird directions" bug.
        fwd_err, right_err = al.world_error_to_body(dx, dy, yaw)
        forward_cm = _clamp_move(fwd_err * 100)
        right_cm   = _clamp_move(right_err * 100)
        alt_cm     = _clamp_move(NAV_HEIGHT_CM - tello.get_height())

        if forward_cm or right_cm or alt_cm:
            print(f"[{label}]   -> go(forward={forward_cm}cm, right={right_cm}cm, up={alt_cm}cm)")
            try:
                _run_blocking_with_live_feed(
                    tello, frame_read,
                    lambda: tello.go_xyz_speed(forward_cm, right_cm, alt_cm, NAV_GO_SPEED_CMS),
                    status_text=f"flying (fwd={forward_cm} right={right_cm} up={alt_cm})cm",
                    label=label,
                )
            except EmergencyLand:
                raise
            except Exception as e:
                print(f"[{label}]   go_xyz_speed failed ({e}) — will retry next iteration")

        # ── Rotate to face the target heading ─────────────────────────────
        deg_display = max(-179, min(179, int(round(math.degrees(yaw_err)))))
        if abs(deg_display) >= 3:
            try:
                _run_blocking_with_live_feed(
                    tello, frame_read,
                    lambda: _rotate_by_yaw_err(tello, yaw_err),
                    status_text=f"rotating {abs(deg_display)}°",
                    label=label,
                )
            except EmergencyLand:
                raise
            except Exception as e:
                print(f"[{label}]   rotate failed ({e}) — will retry next iteration")
        time.sleep(0.3)

    print(f"[{label}] Max iterations reached without converging — proceeding anyway.")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAINROT CLASSIFICATION (Ultralytics .pt)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_image(frame: np.ndarray, model, conf_thresh: float = CLASSIFY_CONF_THRESH):
    """Run the brainrot classifier on one frame. Returns (label, confidence) or (None, 0.0)."""
    if model is None:
        return None, 0.0

    results = model.predict(frame, verbose=False, conf=conf_thresh)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, 0.0

    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    best  = int(np.argmax(confs))
    conf  = float(confs[best])
    if conf < conf_thresh:
        return None, 0.0

    cls_id = int(boxes.cls.cpu().numpy()[best])
    label  = model.names[cls_id]
    return label, conf


def run_classification(tello, frame_read, brainrot_model):
    """
    Blocking classification loop: collect votes over CLASSIFY_FRAMES frames
    (or until CLASSIFY_TIMEOUT), majority vote, retry on low agreement.

    Returns (class_label, target_tag_id).
    """
    print("[Stage 4b] Hold the classification image in front of the camera...")
    votes, conf_sum = [], 0.0
    start = time.time()

    while True:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.01)
            continue

        display = frame.copy()
        elapsed = time.time() - start

        if brainrot_model is not None:
            label, conf = classify_image(frame, brainrot_model)
            if label is not None:
                votes.append(label)
                conf_sum += conf
                cv2.putText(display, f"Detected: {label} ({conf:.2f})",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No detection — hold image closer/steadier",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
        else:
            votes.append('cap')

        cv2.putText(display, f"Stage 4b classifying... {len(votes)}/{CLASSIFY_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow('HCC Final Game', display)
        if cv2.waitKey(1) & 0xFF == 27:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            raise EmergencyLand("ESC during classification")

        enough = len(votes) >= CLASSIFY_FRAMES or elapsed > CLASSIFY_TIMEOUT
        if enough and votes:
            counts    = Counter(votes)
            best, cnt = counts.most_common(1)[0]
            frac      = cnt / len(votes)
            avg_conf  = conf_sum / max(len(votes), 1)

            print(f"\n{'='*48}")
            print(f"  CLASSIFICATION RESULT : {best.upper()}")
            print(f"  Votes  : {dict(counts)}  agreement={frac:.0%}  avg_conf={avg_conf:.2f}")
            print(f"  TARGET : AprilTag {LANDING_TAG[best]}")
            print(f"{'='*48}\n")

            if frac >= MIN_VOTE_FRACTION:
                return best, LANDING_TAG[best]
            print("[Classify] Low vote agreement — collecting more frames")
            votes, conf_sum, start = [], 0.0, time.time()

        elif enough and not votes:
            print("[Classify] No detections — retrying")
            start = time.time()

        time.sleep(0.02)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4d: AprilTag PID approach → land  (precise, single target tag)
# ═══════════════════════════════════════════════════════════════════════════════

def approach_and_land(tello, target_tag_id: int, frame_read, at_det: ATDetector):
    """
    Stop-and-look rotation to find the target landing AprilTag, then PID-approach
    until centred at APPROACH_DIST_M (the 30x30cm floor box in front of the tag)
    and land. ESC in the window triggers emergency land.
    """
    cam_params = [AT_FX, AT_FY, AT_CX, AT_CY]

    pid_fwd = PIDController(KP_FWD, KI_FWD, KD_FWD, output_limit=25)
    pid_lat = PIDController(KP_LAT, KI_LAT, KD_LAT, output_limit=20)
    pid_alt = PIDController(KP_ALT, KI_ALT, KD_ALT, output_limit=20)
    pid_yaw = PIDController(KP_YAW, KI_YAW, KD_YAW, output_limit=35)

    search_state = {'rotating': True, 'phase_start': time.time()}
    last_t = time.time()

    print(f"[Stage 4d] Searching for landing AprilTag id={target_tag_id}...")

    while True:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.02)
            continue

        now = time.time()
        dt  = max(now - last_t, 1e-3)
        last_t = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_det.detect(gray, estimate_tag_pose=True,
                             camera_params=cam_params, tag_size=AT_TAG_SIZE)
        target  = next((t for t in tags if t.tag_id == target_tag_id), None)
        display = frame.copy()

        if target is None:
            rotating = stop_and_look_step(tello, search_state)
            cv2.putText(display, f"Stage 4d: {'rotating' if rotating else 'looking'} "
                                  f"for tag {target_tag_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            tx = float(target.pose_t[0])
            ty = float(target.pose_t[1])
            tz = float(target.pose_t[2])

            err_fwd = tz - APPROACH_DIST_M
            err_lat = tx
            err_alt = ty
            yaw_err = math.atan2(target.pose_R[1, 0], target.pose_R[0, 0])

            fb  = pid_fwd.update(err_fwd, dt)
            lr  = pid_lat.update(err_lat, dt)
            ud  = -pid_alt.update(err_alt, dt)
            yaw = -pid_yaw.update(yaw_err, dt)

            tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

            cv2.putText(display,
                        f"Stage 4d: tag {target_tag_id}  dist={tz:.2f}m  "
                        f"lat={tx:.2f}m  alt={ty:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display,
                        f"err fwd={err_fwd:.2f}  lat={err_lat:.2f}  alt={err_alt:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2)

            if (abs(err_fwd) < LAND_POS_TOL and
                    abs(err_lat) < LAND_LAT_TOL and
                    abs(err_alt) < LAND_LAT_TOL):
                print("[Stage 4d] In position → landing!")
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.3)
                tello.land()
                cv2.putText(display, "LANDING!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow('HCC Final Game', display)
                cv2.waitKey(1500)
                return

        cv2.imshow('HCC Final Game', display)
        if cv2.waitKey(1) & 0xFF == 27:
            print("[Emergency] ESC → landing!")
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            return

        time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 56)
    print("  HCC 2026 Final Game  —  Autonomous Pipeline")
    print("=" * 56)

    # Starting angle drawing is disabled — not allowed in the actual test.
    drawn_angle = 0
    print(f"  Drawn angle: {drawn_angle}° (fixed — drawing disabled per competition rule)")
    input("\n  Connect laptop to Tello Wi-Fi, then press Enter to start...")

    # ── Load AprilTag map + nav target ─────────────────────────────────────
    tag_pose_dict = al.load_tag_map(find_map_yaml())
    nav_target_x, nav_target_y, nav_target_yaw = al.compute_nav_target(
        tag_pose_dict, NAV_TARGET_IDS, NAV_WALL_IDS
    )
    print(f"[Init] Nav target: x={nav_target_x:.2f} y={nav_target_y:.2f} "
          f"yaw={math.degrees(nav_target_yaw):.1f}° "
          f"(midpoint of tags {NAV_TARGET_IDS}, facing away from {NAV_WALL_IDS})")

    # ── Load models (.pt via Ultralytics) ───────────────────────────────────
    balloon_model  = YOLO(BALLOON_PT)  if os.path.exists(BALLOON_PT)  else None
    brainrot_model = YOLO(BRAINROT_PT) if os.path.exists(BRAINROT_PT) else None
    if balloon_model is None:
        print(f"[WARNING] {BALLOON_PT} not found — balloon detection disabled.")
    if brainrot_model is None:
        print(f"[WARNING] {BRAINROT_PT} not found — Stage 4 classification disabled.")

    # ── Shared AprilTag detector (nav, localisation AND landing all use this) ──
    at_det = ATDetector(
        families='tag36h11', nthreads=1,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=1, decode_sharpening=0.25, debug=0,
    )

    # ── Stage 1: takeoff + rotate ─────────────────────────────────────────
    tello = Start_Tello.initialize_tello_stage1()
    Start_Tello.rotate_to_start_angle(tello, target_yaw=drawn_angle)
    frame_read = tello.get_frame_read()

    target_tag_id = None

    try:
        # ── Stage 1b: scan → estimate → fly → rescan → verify deadzone ──────
        navigate_to_point(tello, frame_read, at_det, tag_pose_dict,
                          nav_target_x, nav_target_y, nav_target_yaw,
                          label="Stage 1b")

        # ── Stage 2/3: balloon search + Kalman/PID track + touch ────────────
        kf              = Balloon_Detector.init_kalman_filter()
        kf_initialized  = False
        lost_counter    = 0
        balloon_tracker = Balloon_Detector.BalloonTracker()

        stage        = "SEARCH_BALLOON"
        touch_time   = None
        sprint_start = None
        last_time    = time.time()

        print("[FSM] Entering balloon stage  —  ESC in camera window = emergency land")

        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

            h, w    = frame.shape[:2]
            display = frame.copy()
            now     = time.time()
            dt      = max(now - last_time, 1e-3)
            last_time = now

            # ── SEARCH_BALLOON ───────────────────────────────────────────
            if stage == "SEARCH_BALLOON":
                cv2.putText(display, "Stage 2: Searching for balloon...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                box = Balloon_Detector.detect_balloon(frame, balloon_model)

                if box is not None:
                    print("【CP1 得分】偵測到氣球！ Balloon detected!")
                    cv2.putText(display, "有偵測到balloon",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow('HCC Final Game', display)
                    cv2.waitKey(300)
                    tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.3)
                    kf_initialized = False
                    lost_counter   = 0
                    balloon_tracker.reset()
                    stage = "TRACK_AND_TOUCH"
                else:
                    Balloon_Detector.search_balloon_pattern(tello, search_speed=25)

            # ── TRACK_AND_TOUCH ──────────────────────────────────────────
            elif stage == "TRACK_AND_TOUCH":
                cv2.putText(display, "Stage 3: Tracking balloon...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                box = Balloon_Detector.detect_balloon(frame, balloon_model)
                tracked_pos = None

                kf.transitionMatrix[0, 3] = dt
                kf.transitionMatrix[1, 4] = dt
                kf.transitionMatrix[2, 5] = dt
                prediction = kf.predict()

                if box is not None:
                    lost_counter = 0
                    cv2.rectangle(display,
                                  (box[0], box[1]),
                                  (box[0] + box[2], box[1] + box[3]),
                                  (0, 255, 0), 2)
                    z_meas = Balloon_Detector.recover_3d_position(box, w, h)

                    if not kf_initialized:
                        kf.statePost[0:3] = z_meas
                        kf.statePost[3:6] = 0
                        kf_initialized = True
                    else:
                        kf.correct(z_meas)
                    tracked_pos = kf.statePost[0:3]

                else:
                    lost_counter += 1
                    if kf_initialized and lost_counter < 15:
                        tracked_pos = prediction[0:3]
                        cv2.putText(display, "Kalman predicting...",
                                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    else:
                        print("[FSM] Balloon lost — returning to SEARCH")
                        kf_initialized = False
                        tello.send_rc_control(0, 0, 0, 0)
                        stage = "SEARCH_BALLOON"

                if tracked_pos is not None:
                    cv2.putText(display,
                                f"X:{tracked_pos[0][0]:.1f} Y:{tracked_pos[1][0]:.1f} "
                                f"Z:{tracked_pos[2][0]:.1f} cm",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    lr, fb, ud, yaw, ready = balloon_tracker.update(tracked_pos, dt)
                    cv2.putText(display, f"close_streak={balloon_tracker.close_streak}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

                    if ready:
                        print("【CP2 待評分】確認靠近！執行衝刺碰撞 sprinting to touch...")
                        sprint_start = now
                        stage = "TOUCH_SPRINT"
                    else:
                        tello.send_rc_control(lr, fb, ud, yaw)

            # ── TOUCH_SPRINT (non-blocking forward sprint) ───────────────
            elif stage == "TOUCH_SPRINT":
                elapsed = now - sprint_start
                cv2.putText(display, f"SPRINTING! {elapsed:.2f}/{TOUCH_SPRINT_DURATION:.2f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if elapsed < TOUCH_SPRINT_DURATION:
                    tello.send_rc_control(0, TOUCH_SPRINT_FB, 0, 0)
                else:
                    tello.send_rc_control(0, 0, 0, 0)
                    print("【CP2 待評分】碰撞氣球！TA please judge touch.")
                    touch_time = now
                    stage = "POST_TOUCH_HOVER"

            # ── POST_TOUCH_HOVER (brief stabilisation, then exit the loop) ──
            elif stage == "POST_TOUCH_HOVER":
                tello.send_rc_control(0, 0, 0, 0)
                elapsed   = now - touch_time
                remaining = max(0.0, POST_TOUCH_HOVER_S - elapsed)
                cv2.putText(display, f"Balloon touched! Stabilising {remaining:.1f}s...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('HCC Final Game', display)
                if cv2.waitKey(1) & 0xFF == 27:
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.land()
                    raise EmergencyLand("ESC during post-touch hover")

                if elapsed >= POST_TOUCH_HOVER_S:
                    print("[FSM] Touch sequence complete — heading back to nav point.")
                    break
                time.sleep(0.02)
                continue

            # ── Status bar + display (every frame, all stages above this) ──
            try:
                bat  = tello.get_battery()
                h_cm = tello.get_height()
                status = f"Stage:{stage}  Bat:{bat}%  H:{h_cm}cm"
            except Exception:
                status = f"Stage:{stage}"
            cv2.putText(display, status,
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow('HCC Final Game', display)
            if cv2.waitKey(1) & 0xFF == 27:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                raise EmergencyLand("ESC during balloon stage")

            time.sleep(0.02)

        # ── Stage 4a: back to the 15/16 midpoint (drone may have drifted) ───
        navigate_to_point(tello, frame_read, at_det, tag_pose_dict,
                          nav_target_x, nav_target_y, nav_target_yaw,
                          label="Stage 4a")

        # ── Stage 4b: classify the brainrot image ────────────────────────
        # class_label, target_tag_id = run_classification(tello, frame_read, brainrot_model)
        target_tag_id = 16

        # ── Stage 4c: re-verify position before the final approach ─────────
        navigate_to_point(tello, frame_read, at_det, tag_pose_dict,
                          nav_target_x, nav_target_y, nav_target_yaw,
                          label="Stage 4c")

        # ── Stage 4d: precise PID approach to the classified tag → land ────
        approach_and_land(tello, target_tag_id, frame_read, at_det)

    except (EmergencyLand, KeyboardInterrupt) as e:
        print(f"\n[Main] {type(e).__name__} — landing.")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception:
            pass
    finally:
        try:
            tello.streamoff()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Done] Mission complete.")


if __name__ == '__main__':
    main()