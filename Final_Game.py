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
  [Stage 4c]  Re-run the scan→fly→rescan loop ONE more time (drift check)
  [Stage 4d]  Scan→fly→rescan to a standoff point directly in front of the
              CLASSIFIED landing tag (tight deadzone, to land inside the
              30x30cm marked box) → land.

Fixes in this revision
-------------------------
  1. Altitude is no longer part of the one-shot "go" maneuver. Folding a
     height correction into go_xyz_speed's z argument caused large,
     compounding climbs ("flies too high") whenever the height reading
     was off, AND wasn't continuously active. Altitude is now held by its
     own PIDController, running continuously (every frame) during the
     stop-and-look scan phase — active the whole time, never lumped into
     a single ballistic jump.
  2. After every go_xyz_speed/rotate command, the code now waits and
     discards a few frames before trusting the next AprilTag detection.
     The Tello's video pipeline lags during movement commands (a known
     djitellopy issue — frames keep arriving from before the move
     finished), so without this, the very next "scan" would localize off
     a stale frame, compute another large correction on top of a move
     that hadn't actually been reflected yet, and the drone would drift
     in what looks like "weird directions." This also addresses the
     apparent stream pause during these maneuvers.
  3. The final landing approach no longer just lands the moment the
     target tag is glimpsed. It now computes a world-frame standoff point
     directly in front of the SPECIFIC classified tag (verified by
     round-trip simulation) and reuses the same scan→fly→rescan routine,
     restricted to that one tag id, with a tight deadzone sized to land
     inside the 30x30cm marked box — instead of a separate continuous-PID
     approach that could trigger immediately on a noisy first detection.

Detection backend
------------------
Both balloon and brainrot models load from .pt weights via Ultralytics.
  pip install ultralytics djitellopy pupil-apriltags opencv-python scipy pyyaml
"""

import math
import os
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

# ── Scan-fly-rescan navigation (used for the 15/16-midpoint passes) ─────────
NAV_TARGET_IDS  = (15, 16)              # fly to the midpoint of these tags
NAV_WALL_IDS    = (4, 5, 6, 7, 8, 9)    # face away from this wall
NAV_HEIGHT_CM   = 115                  # cruise height during navigation

# Deadzone — intentionally large: this stage just needs to get the drone
# roughly back to the open area facing the right way, not landing-precise.
NAV_POS_TOL_M   = 0.35
NAV_YAW_TOL_RAD = math.radians(20)

NAV_GO_SPEED_CMS   = 40    # speed for the one-shot relative "go" maneuver (xy only) 30 if bugged
NAV_MIN_MOVE_CM    = 10    # Tello SDK requires each axis to be 0 or >= this
NAV_MAX_MOVE_CM    = 300   # safety cap per single move, per axis
NAV_MAX_ITERATIONS = 1     # give up (and proceed anyway) after this many cycles
NAV_SCAN_WARN_S    = 8.0   # print a reminder if no tag found within this long

# A single AprilTag pose estimate is noisy. Once a tag is first spotted, hold
# still and collect several readings, then use their (circular-mean-aware)
# average for the move decision instead of trusting one frame.
NAV_POSE_SAMPLES        = 10
NAV_POSE_COLLECT_TIMEOUT = 6.0   # give up collecting (use whatever we have) after this long

# Altitude is held by its OWN continuous PID during scanning (not part of
# the ballistic "go" maneuver — see fix #1 above).
NAV_ALT_GAINS = (0.6, 0.0, 0.1)   # (kp, ki, kd) on (NAV_HEIGHT_CM - get_height())
NAV_ALT_LIMIT = 25

# Video lags behind movement commands — settle + discard a few frames before
# trusting the next detection (see fix #2 above).
NAV_SETTLE_S      = 1.0
NAV_FLUSH_FRAMES  = 5

# ── Final landing approach (Stage 4d): standoff point in front of ONE tag ──
LANDING_APPROACH_DIST_M = 0.55     # standoff distance — middle of the 40-70cm spec
LANDING_POS_TOL_M       = 0.15     # tight: must land inside the 30x30cm box
LANDING_YAW_TOL_RAD     = math.radians(20)
LANDING_MAX_ITERATIONS  = 1

# ── Classification ────────────────────────────────────────────────────────────
LANDING_TAG          = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}
CLASSIFY_FRAMES      = 50
MIN_VOTE_FRACTION    = 0.50
CLASSIFY_TIMEOUT     = 30.0
CLASSIFY_CONF_THRESH = 0.3

# ── Shared stop-and-look search pattern (continuous yaw blurs AprilTag detection) ──
SEARCH_YAW_PCT  = 40
SEARCH_ROTATE_S = 0.4
SEARCH_PAUSE_S  = 0.6

# ── Touch sprint (non-blocking stage, see TOUCH_SPRINT below) ──
TOUCH_SPRINT_FB       = 70
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


def stop_and_look_step(tello, search_state: dict, ud: int = 0):
    """
    Continuous yaw rotation blurs frames enough that pupil_apriltags often
    misses tags entirely. Alternate short rotation bursts with still pauses
    so the detector gets a sharp frame to work with.

    `ud` lets the caller fold in a continuously-active altitude correction
    (e.g. from a PIDController) without it being a separate, competing
    send_rc_control call.

    search_state must have keys 'rotating' (bool) and 'phase_start' (float),
    mutated in place. Call this once per frame whenever no tag is visible.
    """
    now     = time.time()
    elapsed = now - search_state['phase_start']

    if search_state['rotating']:
        tello.send_rc_control(0, 0, ud, SEARCH_YAW_PCT)
        if elapsed > SEARCH_ROTATE_S:
            tello.send_rc_control(0, 0, 0, 0)
            search_state['rotating']    = False
            search_state['phase_start'] = now
    else:
        tello.send_rc_control(0, 0, ud, 0)
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


def _settle_and_flush(frame_read, settle_time: float = NAV_SETTLE_S, flush_frames: int = NAV_FLUSH_FRAMES):
    """
    After a blocking maneuver, the Tello's video pipeline is typically still
    behind (a known djitellopy/Tello quirk — frames from before the move
    finished keep arriving for a bit). Wait, then pull and discard a few
    frames, so the NEXT detection is from a frame that actually reflects
    where the drone is now.
    """
    time.sleep(settle_time)
    for _ in range(flush_frames):
        _ = frame_read.frame
        time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAN → ESTIMATE → FLY DIRECTLY
# ═══════════════════════════════════════════════════════════════════════════════

def navigate_to_point(tello, frame_read, at_det: ATDetector, tag_pose_dict: dict,
                      target_x: float, target_y: float, target_yaw: float,
                      pos_tol: float = NAV_POS_TOL_M, yaw_tol: float = NAV_YAW_TOL_RAD,
                      max_iterations: int = NAV_MAX_ITERATIONS, label: str = "Nav",
                      allowed_tag_ids=None):
    """
    Repeatedly: scan for a tag (stop-and-look, with continuous altitude
    PID active the whole time), then HOLD STILL and collect NAV_POSE_SAMPLES
    pose estimates from that spot and average them (a single AprilTag pose
    reading is noisy — circular-mean-aware averaging smooths that out, see
    apriltag_localizer.average_poses). Fly DIRECTLY toward the target with
    one relative 'go' maneuver based on the averaged pose (XY only —
    altitude is handled separately, see above), settle + flush stale video
    frames, then repeat. Loops until within (pos_tol, yaw_tol) or
    max_iterations is reached.

    allowed_tag_ids : optional iterable of tag ids. If given, only those
    tags are trusted for localization this call (e.g. lock onto ONE
    specific landing tag instead of "any" tag in the map).
    """
    cam_params  = [AT_FX, AT_FY, AT_CX, AT_CY]
    pid_alt     = PIDController(*NAV_ALT_GAINS, output_limit=NAV_ALT_LIMIT)
    allowed_set = set(allowed_tag_ids) if allowed_tag_ids is not None else None

    print(f"[{label}] Navigating to ({target_x:.2f}, {target_y:.2f}) "
          f"yaw={math.degrees(target_yaw):.1f}°  "
          f"(deadzone: {pos_tol:.2f}m / {math.degrees(yaw_tol):.1f}°)")

    for iteration in range(1, max_iterations + 1):
        # ── Scan for a tag, then HOLD STILL and collect several pose samples ──
        # (a single AprilTag pose estimate is noisy — average several readings
        # taken from the same spot instead of acting on one frame)
        search_state     = {'rotating': True, 'phase_start': time.time()}
        scan_start       = time.time()
        last_t           = time.time()
        warned           = False
        samples          = []
        collect_deadline = None

        while len(samples) < NAV_POSE_SAMPLES:
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
            if allowed_set is not None:
                tags = [t for t in tags if t.tag_id in allowed_set]
            one_pose = al.localize_best_tag(tags, tag_pose_dict)

            alt_err = NAV_HEIGHT_CM - tello.get_height()
            ud      = int(pid_alt.update(alt_err, dt))
            display = frame.copy()

            if one_pose is not None:
                samples.append(one_pose)
                if collect_deadline is None:
                    collect_deadline = time.time() + NAV_POSE_COLLECT_TIMEOUT
                tello.send_rc_control(0, 0, ud, 0)   # hold still, just hold altitude
                cv2.putText(display,
                            f"[{label}] iter {iteration}/{max_iterations}: "
                            f"collecting pose samples {len(samples)}/{NAV_POSE_SAMPLES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)

            elif collect_deadline is not None:
                # Already started collecting — tag flickered out briefly. Hold
                # still and keep trying rather than throwing away good samples.
                tello.send_rc_control(0, 0, ud, 0)
                cv2.putText(display,
                            f"[{label}] iter {iteration}/{max_iterations}: "
                            f"tag flickered, holding ({len(samples)}/{NAV_POSE_SAMPLES} so far)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                if time.time() > collect_deadline:
                    print(f"[{label}] Lost tag while collecting — "
                          f"using {len(samples)} sample(s) gathered so far")
                    break

            else:
                # Haven't found a tag at all yet — scan.
                rotating = stop_and_look_step(tello, search_state, ud=ud)
                cv2.putText(display,
                            f"[{label}] iter {iteration}/{max_iterations}: "
                            f"{'rotating' if rotating else 'looking'} for AprilTag... (ud={ud})",
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

        if not samples:
            print(f"[{label}] No usable pose samples this iteration — retrying.")
            continue

        pose = al.average_poses(samples)
        print(f"[{label}] Averaged {len(samples)} pose sample(s).")
        x, y, z, yaw = pose
        dx, dy  = target_x - x, target_y - y
        pos_err = math.hypot(dx, dy)
        yaw_err = al.wrap_angle(target_yaw - yaw)

        print(f"[{label}] iter {iteration}: pose=({x:.2f},{y:.2f},yaw={math.degrees(yaw):.1f}°)  "
              f"pos_err={pos_err:.2f}m  yaw_err={math.degrees(yaw_err):.1f}°")

        if pos_err < pos_tol and abs(yaw_err) < yaw_tol:
            print(f"[{label}] Within deadzone — done.")
            return True

        # ── Fly DIRECTLY to the estimated point (XY only — one-shot relative 'go') ──
        fwd_err, right_err = al.world_error_to_body(dx, dy, yaw)
        forward_cm = _clamp_move(fwd_err * 100)
        left_cm    = _clamp_move(-right_err * 100)

        if forward_cm or left_cm:
            print(f"[{label}]   -> go(forward={forward_cm}cm, left={left_cm}cm)")
            try:
                tello.go_xyz_speed(forward_cm, left_cm, 0, NAV_GO_SPEED_CMS)
            except Exception as e:
                print(f"[{label}]   go_xyz_speed failed ({e}) — will retry next iteration")

        # ── Rotate to face the target heading ─────────────────────────────
        _rotate_by_yaw_err(tello, yaw_err)

        # ── Let the video pipeline catch up before trusting the next scan ──
        _settle_and_flush(frame_read)

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

        stage        = "SEARCH_BALLOON" #"SEARCH_BALLOON"
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
                    Balloon_Detector.search_balloon_pattern(tello, search_speed=-35)

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
                    tello.send_rc_control(0, TOUCH_SPRINT_FB, -25, 0)
                else:
                    tello.send_rc_control(0, 0, 0, 0)
                    print("【CP2 待評分】碰撞氣球！TA please judge touch.")
                    touch_time = now
                    stage = "POST_TOUCH_HOVER"

            # ── POST_TOUCH_HOVER (brief stabilisation, then exit the loop) ──
            elif stage == "POST_TOUCH_HOVER":
                global NAV_HEIGHT_CM
                NAV_HEIGHT_CM = 135
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
        class_label, target_tag_id = run_classification(tello, frame_read, brainrot_model)
        if(target_tag_id == 15): 
            global NAV_HEIGHT_CM
            NAV_HEIGHT_CM = 150

        # tello.rotate_clockwise(180)
        # # ── Stage 4c: re-verify position before the final approach ─────────
        # navigate_to_point(tello, frame_read, at_det, tag_pose_dict,
        #                   nav_target_x, nav_target_y, nav_target_yaw,
        #                   label="Stage 4c")

        tello.rotate_clockwise(90)
        tello.send_rc_control(0, -80, 0, 0)
        # ── Stage 4d: fly to the standoff point in front of the CLASSIFIED
        #              tag (tight deadzone, sized for the 30x30cm box) → land ──
        target_entry = tag_pose_dict[target_tag_id]
        land_x, land_y, land_yaw = al.compute_landing_standoff(target_entry, LANDING_APPROACH_DIST_M)
        print(f"[Stage 4d] Standoff point for tag {target_tag_id}: "
              f"({land_x:.2f}, {land_y:.2f}) yaw={math.degrees(land_yaw):.1f}°")

        navigate_to_point(tello, frame_read, at_det, tag_pose_dict,
                          land_x, land_y, land_yaw,
                          pos_tol=LANDING_POS_TOL_M, yaw_tol=LANDING_YAW_TOL_RAD,
                          max_iterations=LANDING_MAX_ITERATIONS, label="Stage 4d",
                          allowed_tag_ids={target_tag_id})

        print("[Stage 4d] In position — landing!")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.3)
        tello.land()

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