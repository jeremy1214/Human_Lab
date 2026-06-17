#!/usr/bin/env python3
"""
Final_Game.py  —  HCC 2026 Autonomous Competition Pipeline
===========================================================
Run:   python Final_Game.py

No keyboard input is needed after the pre-flight setup question.
Press ESC in the camera window only for emergency land.

Pipeline
--------
  [Setup   ]  Enter drawn yaw angle (before drone connects)
  [Stage 1 ]  Takeoff → rotate to drawn angle
  [Stage 1b]  Self-localize from ANY visible AprilTag, fly to the midpoint
              of tags 15/16, face away from the localization wall (4-9)
  [Stage 2 ]  Search balloon (slow yaw scan)
  [Stage 3 ]  Kalman + PID track → confirmed-close sprint → touch
  [Post    ]  Hover after touch, TA positions image
  [Stage 4 ]  Brainrot classification (majority vote)
  [Stage 4 ]  Rotate (stop-and-look) until landing AprilTag visible
  [Stage 4 ]  PID approach → land

Changes in this revision
--------------------------
  1. NEW: self-localization + navigation stage (apriltag_localizer.py).
     The drone reads its own world position/heading from a single visible
     AprilTag, flies to the midpoint of tags 15 & 16, and turns to face
     away from the 4-9 wall, before starting the balloon search.
  2. FIXED: the balloon "touch" trigger no longer fires off one noisy
     frame. BalloonTracker (Balloon_Detector.py) requires several
     consecutive close+centred frames, and the calibration constants now
     match the camera's real (Lab-1) focal length instead of a guess.
  3. FIXED: every PID controller in the project (balloon tracking AND the
     Stage-4 AprilTag approach) now uses the shared, dt-aware PIDController
     class (pid_controller.py) — implemented to the same spec as pid.py's
     PID.update() TODO — instead of an ad-hoc, dt-blind formula.
  4. The balloon sprint-and-touch maneuver is now a non-blocking stage
     (TOUCH_SPRINT) instead of a blocking time.sleep() buried inside the
     per-frame tracking update.

Detection backend
------------------
Both balloon and brainrot models load from .pt weights via Ultralytics.
  pip install ultralytics djitellopy pupil-apriltags opencv-python scipy pyyaml
"""

import math
import os
import time
from collections import Counter

import apriltag_localizer as al  # Self-localization + nav-target math
import cv2
import numpy as np
from pid_controller import PIDController
from pupil_apriltags import Detector as ATDetector
from ultralytics import YOLO

import Balloon_Detector  # Stage 2/3 logic (Kalman + PID + touch)
import Start_Tello  # Stage 1 takeoff helpers

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

# ── Stage 1b: self-localize & navigate to start point ──────────────────────
NAV_TARGET_IDS  = (15, 16)              # fly to the midpoint of these tags
NAV_WALL_IDS    = (4, 5, 6, 7, 8, 9)    # face away from this wall
NAV_HEIGHT_CM   = 130                   # cruise height during navigation
NAV_POS_TOL_M   = 0.15                  # position tolerance to call it "arrived"
NAV_YAW_TOL_RAD = math.radians(8)       # heading tolerance
NAV_KP_POS, NAV_KI_POS, NAV_KD_POS = 35.0, 0.0, 6.0
NAV_KP_YAW, NAV_KI_YAW, NAV_KD_YAW = 45.0, 0.0, 5.0
NAV_KP_ALT                          = 0.6   # cm error -> RC%, simple P on Tello's own height sensor

# ── Classification ────────────────────────────────────────────────────────────
LANDING_TAG          = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}
CLASSIFY_FRAMES      = 20
MIN_VOTE_FRACTION    = 0.50
CLASSIFY_TIMEOUT     = 30.0
CLASSIFY_CONF_THRESH = 0.3

# ── Stage-4 approach ─────────────────────────────────────────────────────────
APPROACH_DIST_M = 0.55
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

# ── Touch sprint (now its own non-blocking stage, see TOUCH_SPRINT below) ──
TOUCH_SPRINT_FB       = 45    # RC % forward during the sprint
TOUCH_SPRINT_DURATION = 0.8   # seconds

# ── Timing ────────────────────────────────────────────────────────────────────
POST_TOUCH_HOVER_S = 3.0


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


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: AprilTag PID approach → land
# ═══════════════════════════════════════════════════════════════════════════════

def approach_and_land(tello, target_tag_id: int, frame_read, at_det: ATDetector):
    """
    Stop-and-look rotation to find the target landing AprilTag, then PID-approach
    to APPROACH_DIST_M and land. ESC in the window triggers emergency land.
    """
    cam_params = [AT_FX, AT_FY, AT_CX, AT_CY]

    pid_fwd = PIDController(KP_FWD, KI_FWD, KD_FWD, output_limit=25)
    pid_lat = PIDController(KP_LAT, KI_LAT, KD_LAT, output_limit=20)
    pid_alt = PIDController(KP_ALT, KI_ALT, KD_ALT, output_limit=20)
    pid_yaw = PIDController(KP_YAW, KI_YAW, KD_YAW, output_limit=35)

    search_state = {'rotating': True, 'phase_start': time.time()}
    last_t = time.time()

    print(f"[Stage 4] Searching for AprilTag id={target_tag_id}...")

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
            cv2.putText(display, f"Stage 4: {'rotating' if rotating else 'looking'} "
                                  f"for tag {target_tag_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # Camera-frame translation (OpenCV: +X right, +Y down, +Z forward)
            tx = float(target.pose_t[0])
            ty = float(target.pose_t[1])
            tz = float(target.pose_t[2])

            err_fwd = tz - APPROACH_DIST_M
            err_lat = tx
            err_alt = ty
            yaw_err = math.atan2(target.pose_R[1, 0], target.pose_R[0, 0])

            fb  = pid_fwd.update(err_fwd, dt)
            lr  = pid_lat.update(err_lat, dt)     # + → tag right → move right (no extra negation)
            ud  = -pid_alt.update(err_alt, dt)    # tag-below(+) -> Tello up(+) needs negation
            yaw = -pid_yaw.update(yaw_err, dt)    # flip if drone yaws the wrong way on hardware

            tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

            cv2.putText(display,
                        f"Stage 4: tag {target_tag_id}  dist={tz:.2f}m  "
                        f"lat={tx:.2f}m  alt={ty:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display,
                        f"err fwd={err_fwd:.2f}  lat={err_lat:.2f}  alt={err_alt:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2)

            if (abs(err_fwd) < LAND_POS_TOL and
                    abs(err_lat) < LAND_LAT_TOL and
                    abs(err_alt) < LAND_LAT_TOL):
                print("[Stage 4] In position → landing!")
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

    # ── Pre-flight: get drawn angle (before drone connects) ───────────────
    while True:
        try:
            # raw = input("\n  Enter drawn starting angle in degrees (0 / 90 / 180 / 270): ")
            # drawn_angle = int(raw.strip())
            drawn_angle = 0
            print(f"  Drawn angle confirmed: {drawn_angle}° (fixed)")
            break
        except ValueError:
            print("  Please enter an integer (e.g. 90).")
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

    # ── Shared AprilTag detector (used for nav, localisation AND landing) ──
    at_det = ATDetector(
        families='tag36h11', nthreads=1,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=1, decode_sharpening=0.25, debug=0,
    )
    cam_params = [AT_FX, AT_FY, AT_CX, AT_CY]

    # ── Stage 1: takeoff + rotate ─────────────────────────────────────────
    tello = Start_Tello.initialize_tello_stage1()
    Start_Tello.rotate_to_start_angle(tello, target_yaw=drawn_angle)
    frame_read = tello.get_frame_read()

    # ── Stage 1b state: self-localize & navigate ────────────────────────────
    pid_nav_fwd = PIDController(NAV_KP_POS, NAV_KI_POS, NAV_KD_POS, output_limit=30)
    pid_nav_lat = PIDController(NAV_KP_POS, NAV_KI_POS, NAV_KD_POS, output_limit=30)
    pid_nav_yaw = PIDController(NAV_KP_YAW, NAV_KI_YAW, NAV_KD_YAW, output_limit=40)
    nav_search_state = {'rotating': True, 'phase_start': time.time()}
    last_known_pose   = None   # (x, y, z, yaw) — last successful localisation

    # ── Kalman + PID state (Stage 2/3) ────────────────────────────────────
    kf              = Balloon_Detector.init_kalman_filter()
    kf_initialized  = False
    lost_counter    = 0
    balloon_tracker = Balloon_Detector.BalloonTracker()

    # ── Stage state machine (explicit variable initialisation) ────────────
    stage          = "LOCALIZE_AND_NAVIGATE"
    touch_time     = None
    sprint_start   = None
    classify_start = None
    votes          = []
    conf_sum       = 0.0
    class_label    = None
    target_tag_id  = None

    last_time = time.time()
    print("[FSM] Entering competition loop  —  ESC in camera window = emergency land")

    try:
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

            # ── LOCALIZE_AND_NAVIGATE ─────────────────────────────────────
            if stage == "LOCALIZE_AND_NAVIGATE":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tags = at_det.detect(gray, estimate_tag_pose=True,
                                     camera_params=cam_params, tag_size=AT_TAG_SIZE)
                pose = al.localize_best_tag(tags, tag_pose_dict)

                if pose is None:
                    rotating = stop_and_look_step(tello, nav_search_state)
                    cv2.putText(display,
                                f"Stage 1b: {'rotating' if rotating else 'looking'} for any AprilTag...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    last_known_pose = pose
                    x, y, z, yaw = pose

                    dx, dy  = nav_target_x - x, nav_target_y - y
                    fwd_err, right_err = al.world_error_to_body(dx, dy, yaw)
                    yaw_err = al.wrap_angle(nav_target_yaw - yaw)

                    pos_err = math.hypot(dx, dy)

                    if pos_err < NAV_POS_TOL_M and abs(yaw_err) < NAV_YAW_TOL_RAD:
                        print(f"[Stage 1b] Arrived at nav target "
                              f"(pos_err={pos_err:.2f}m yaw_err={math.degrees(yaw_err):.1f}°)")
                        tello.send_rc_control(0, 0, 0, 0)
                        time.sleep(0.3)
                        kf_initialized = False
                        lost_counter   = 0
                        balloon_tracker.reset()
                        stage = "SEARCH_BALLOON"
                    else:
                        fb  = pid_nav_fwd.update(fwd_err, dt)
                        lr  = pid_nav_lat.update(right_err, dt)
                        yaw_cmd = -pid_nav_yaw.update(yaw_err, dt)   # CCW+ world -> Tello CW+ needs negation

                        alt_err = NAV_HEIGHT_CM - tello.get_height()
                        ud = int(np.clip(NAV_KP_ALT * alt_err, -25, 25))

                        tello.send_rc_control(int(lr), int(fb), ud, int(yaw_cmd))

                        cv2.putText(display,
                                    f"Stage 1b: nav  pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}d  "
                                    f"err={pos_err:.2f}m  yaw_err={math.degrees(yaw_err):.1f}d",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
                        cv2.putText(display,
                                    f"target=({nav_target_x:.2f},{nav_target_y:.2f}) "
                                    f"yaw={math.degrees(nav_target_yaw):.1f}d",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            # ── SEARCH_BALLOON ───────────────────────────────────────────
            elif stage == "SEARCH_BALLOON":
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

            # ── POST_TOUCH_HOVER ─────────────────────────────────────────
            elif stage == "POST_TOUCH_HOVER":
                tello.send_rc_control(0, 0, 0, 0)
                elapsed   = now - touch_time
                remaining = max(0.0, POST_TOUCH_HOVER_S - elapsed)
                cv2.putText(display, f"Balloon touched!  Hovering {remaining:.1f}s...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display,
                            "TA: please hold classification image in front of camera",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

                if elapsed >= POST_TOUCH_HOVER_S:
                    print("[FSM] Starting image classification...")
                    votes          = []
                    conf_sum       = 0.0
                    classify_start = now
                    stage = "CLASSIFY"

            # ── CLASSIFY ─────────────────────────────────────────────────
            elif stage == "CLASSIFY":
                tello.send_rc_control(0, 0, 0, 0)
                elapsed = now - classify_start

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

                cv2.putText(display,
                            f"Stage 4 classifying... {len(votes)}/{CLASSIFY_FRAMES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                enough = len(votes) >= CLASSIFY_FRAMES or elapsed > CLASSIFY_TIMEOUT
                if enough and votes:
                    counts    = Counter(votes)
                    best, cnt = counts.most_common(1)[0]
                    frac      = cnt / len(votes)
                    avg_conf  = conf_sum / max(len(votes), 1)

                    print(f"\n{'='*48}")
                    print(f"  CLASSIFICATION RESULT : {best.upper()}")
                    print(f"  Votes  : {dict(counts)}  "
                          f"agreement={frac:.0%}  avg_conf={avg_conf:.2f}")
                    print(f"  TARGET : AprilTag {LANDING_TAG[best]}")
                    print(f"{'='*48}\n")

                    if frac >= MIN_VOTE_FRACTION:
                        class_label   = best
                        target_tag_id = LANDING_TAG[best]
                        stage = "NAVIGATE"
                    else:
                        print("[Classify] Low vote agreement — collecting more frames")
                        votes          = []
                        conf_sum       = 0.0
                        classify_start = now

                elif enough and not votes:
                    print("[Classify] No detections — retrying")
                    classify_start = now

            # ── NAVIGATE (break to approach loop) ────────────────────────
            elif stage == "NAVIGATE":
                cv2.putText(display,
                            f"Classified: {class_label}  →  Tag {target_tag_id}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
                cv2.imshow('HCC Final Game', display)
                cv2.waitKey(800)
                break

            # ── Status bar (every frame) ──────────────────────────────────
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
                print("[Emergency] ESC pressed → emergency land!")
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                tello.streamoff()
                cv2.destroyAllWindows()
                return

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl-C → landing")
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()
        return

    # ── Stage 4 approach + land ───────────────────────────────────────────
    try:
        approach_and_land(tello, target_tag_id, frame_read, at_det)
    except Exception as e:
        print(f"[Stage 4] Exception: {e} — emergency land")
        tello.send_rc_control(0, 0, 0, 0)
        try:
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