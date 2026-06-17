#!/usr/bin/env python3
"""
Final_Game.py  —  HCC 2026 Autonomous Competition Pipeline
===========================================================
Run:   python Final_Game.py

No keyboard input is needed after the pre-flight setup question.
Press ESC in the camera window only for emergency land.

Pipeline
--------
  [Setup ]  Enter drawn yaw angle (before drone connects)
  [Stage 1] Takeoff → rotate to drawn angle
  [Stage 2] Search balloon (slow yaw scan)
  [Stage 3] Kalman + PID track → sprint → touch
  [Post  ]  Hover after touch, TA positions image
  [Stage 4] Brainrot classification (majority vote)
  [Stage 4] Rotate (stop-and-look) until landing AprilTag visible
  [Stage 4] PID approach → land

Detection backend
------------------
Both balloon and brainrot models now load directly from .pt weights via
Ultralytics — no ONNX conversion needed.
  pip install ultralytics djitellopy pupil-apriltags opencv-python scipy

Changes from the previous version
-----------------------------------
  1. Switched balloon + brainrot detection from .onnx (onnxruntime / cv2.dnn)
     to .pt (Ultralytics YOLO) — see detect_balloon() and classify_image().
  2. AprilTag Stage-4 approach was diverging instead of converging: the
     lateral PID term had its sign flipped (verified analytically and by
     simulation). Fixed: lr = +KP_LAT * err_lat (was negated).
  3. Stage-4 search now uses a stop-and-look rotation pattern instead of
     continuous yaw — continuous rotation was blurring frames badly enough
     that pupil_apriltags frequently failed to detect the tag at all.
  4. Unused ROS-era / parallel-architecture files (main.py, image_classifier.py,
     apriltag_detector.py, control_tello_ekf.py, ekf_localization.py) are no
     longer imported anywhere — delete them, this file is self-contained.
"""

import math
import os
import time
from collections import Counter

import cv2
import numpy as np
from pupil_apriltags import Detector as ATDetector
from ultralytics import YOLO

import Balloon_Detector  # Stage 2 / 3 logic (Kalman + PID + touch)
import Start_Tello  # Stage 1 takeoff helpers

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  tune these numbers before the competition
# ═══════════════════════════════════════════════════════════════════════════════

BALLOON_PT  = 'balloon.pt'          # Ultralytics balloon detector
BRAINROT_PT = 'brainrot_detect.pt'  # Ultralytics brainrot classifier

# ── AprilTag camera intrinsics (Lab 1 calibration) ──────────────────────────
AT_FX, AT_FY    = 835.342103847164, 839.4691450667409
AT_CX, AT_CY    = 415.5366635247159, 355.11975613817964

# Physical AprilTag side length in metres.
# NOTE: verify this matches the *printed* landing tags (13-16) — they may be
# a different physical size than the localisation tags (4-12) used in the
# ROS map. If the approach distance reads consistently wrong on real hardware
# (e.g. always too close or too far), this is the first thing to check.
AT_LANDING_TAG_SIZE = 0.165   # metres

# ── Classification ────────────────────────────────────────────────────────────
LANDING_TAG       = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}
CLASSIFY_FRAMES   = 20    # vote-collection window
MIN_VOTE_FRACTION = 0.50  # majority needed
CLASSIFY_TIMEOUT  = 30.0  # seconds; reset & retry if inconclusive
CLASSIFY_CONF_THRESH = 0.15

# ── Stage-4 approach ─────────────────────────────────────────────────────────
APPROACH_DIST_M = 0.55    # metres in front of tag (40-70 cm spec)
LAND_POS_TOL    = 0.10    # forward-error tolerance (m) to trigger land
LAND_LAT_TOL    = 0.08    # lateral / vertical tolerance (m)

# Stop-and-look search pattern (continuous rotation blurs AprilTag detection)
SEARCH_YAW_PCT   = 25     # RC % while actively rotating
SEARCH_ROTATE_S  = 0.4    # seconds of rotation per cycle
SEARCH_PAUSE_S   = 0.6    # seconds held still per cycle (let the image settle)

# Proportional gains (output in RC %, clamped to limit below)
KP_FWD, KP_LAT, KP_ALT, KP_YAW = 40.0, 50.0, 40.0, 50.0

# ── Timing ────────────────────────────────────────────────────────────────────
POST_TOUCH_HOVER_S = 3.0  # hover after touch so TA can walk up with image


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAINROT CLASSIFICATION (Ultralytics .pt)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_image(frame: np.ndarray, model, conf_thresh: float = CLASSIFY_CONF_THRESH):
    """
    Classify a single frame with the Ultralytics brainrot model.

    Returns (label: str, confidence: float) or (None, 0.0) if nothing detected.
    """
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

def _rc_clamp(val, limit):
    return int(np.clip(val, -limit, limit))


def approach_and_land(tello, target_tag_id: int, frame_read):
    """
    Stop-and-look rotation to find the target landing AprilTag, then PID-approach
    to APPROACH_DIST_M and land.  ESC in the window triggers emergency land.
    """
    at_det = ATDetector(
        families='tag36h11', nthreads=1,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=1, decode_sharpening=0.25, debug=0,
    )
    cam_params = [AT_FX, AT_FY, AT_CX, AT_CY]

    print(f"[Stage 4] Searching for AprilTag id={target_tag_id}...")

    # Continuous rotation blurs frames enough that pupil_apriltags frequently
    # misses the tag — alternate short rotation bursts with still pauses.
    rotating    = True
    phase_start = time.time()

    while True:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.02)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_det.detect(
            gray, estimate_tag_pose=True,
            camera_params=cam_params, tag_size=AT_LANDING_TAG_SIZE,
        )
        target  = next((t for t in tags if t.tag_id == target_tag_id), None)
        display = frame.copy()

        if target is None:
            now     = time.time()
            elapsed = now - phase_start

            if rotating:
                tello.send_rc_control(0, 0, 0, SEARCH_YAW_PCT)
                if elapsed > SEARCH_ROTATE_S:
                    tello.send_rc_control(0, 0, 0, 0)
                    rotating    = False
                    phase_start = now
            else:
                tello.send_rc_control(0, 0, 0, 0)   # hold still for a sharp frame
                if elapsed > SEARCH_PAUSE_S:
                    rotating    = True
                    phase_start = now

            cv2.putText(display,
                        f"Stage 4: {'rotating' if rotating else 'looking'} "
                        f"for tag {target_tag_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        else:
            # Camera-frame translation (OpenCV: +X right, +Y down, +Z forward)
            tx = float(target.pose_t[0])
            ty = float(target.pose_t[1])
            tz = float(target.pose_t[2])

            err_fwd = tz - APPROACH_DIST_M   # + → still too far
            err_lat = tx                       # + → tag right of camera
            err_alt = ty                       # + → tag below camera centre
            yaw_err = math.atan2(target.pose_R[1, 0], target.pose_R[0, 0])

            # ── PID → RC percentages ──────────────────────────────────────
            # fb : err_fwd>0 (too far)      → move forward (fb+)
            # lr : err_lat>0 (tag to right) → move RIGHT (lr+) to close the gap.
            #      BUG FIX: this term was previously negated, which sent the
            #      drone left instead — verified by simulation to diverge
            #      exponentially rather than converge. Do not re-add the minus.
            # ud : err_alt>0 (tag below)    → move up (ud+) is achieved by
            #      negating, since Tello's up_down convention is up-positive
            #      while OpenCV's y-axis is down-positive.
            # yaw: heuristic alignment; if the drone yaws away from the tag
            #      instead of squaring up to it, flip this sign.
            fb  = _rc_clamp( KP_FWD * err_fwd,  25)
            lr  = _rc_clamp( KP_LAT * err_lat,  20)
            ud  = _rc_clamp(-KP_ALT * err_alt,  20)
            yaw = _rc_clamp(-KP_YAW * yaw_err,  35)

            tello.send_rc_control(lr, fb, ud, yaw)

            cv2.putText(display,
                        f"Stage 4: tag {target_tag_id}  dist={tz:.2f}m  "
                        f"lat={tx:.2f}m  alt={ty:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display,
                        f"err fwd={err_fwd:.2f}  lat={err_lat:.2f}  alt={err_alt:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2)

            # Landing condition: within tolerance on all axes
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
        if cv2.waitKey(1) & 0xFF == 27:    # ESC
            print("[Emergency] ESC → landing!")
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            return

        time.sleep(0.05)   # ~20 Hz


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
            #raw = input("\n  Enter drawn starting angle in degrees (0 / 90 / 180 / 270): ")
            #drawn_angle = int(raw.strip())
            drawn_angle = 0
            print(f"  Drawn angle confirmed: {drawn_angle}° (fixed)")
            break
        except ValueError:
            print("  Please enter an integer (e.g. 90).")
    input("\n  Connect laptop to Tello Wi-Fi, then press Enter to start...")

    # ── Load models (.pt via Ultralytics) ───────────────────────────────────
    balloon_model  = None
    brainrot_model = None

    if os.path.exists(BALLOON_PT):
        balloon_model = YOLO(BALLOON_PT)
        print(f"[Init] Balloon model loaded ({BALLOON_PT})  classes={balloon_model.names}")
    else:
        print(f"[WARNING] {BALLOON_PT} not found — balloon detection disabled.")

    if os.path.exists(BRAINROT_PT):
        brainrot_model = YOLO(BRAINROT_PT)
        print(f"[Init] Brainrot model loaded ({BRAINROT_PT})  classes={brainrot_model.names}")
    else:
        print(f"[WARNING] {BRAINROT_PT} not found — Stage 4 classification disabled.")

    # ── Stage 1: takeoff + rotate ─────────────────────────────────────────
    tello = Start_Tello.initialize_tello_stage1()
    Start_Tello.rotate_to_start_angle(tello, target_yaw=drawn_angle)
    frame_read = tello.get_frame_read()

    # ── Kalman + PID state ────────────────────────────────────────────────
    kf             = Balloon_Detector.init_kalman_filter()
    kf_initialized = False
    lost_counter   = 0
    pid_states     = (0, 0, 0, 0, 0, 0)
    last_time      = time.time()

    # ── Stage state machine (explicit variable initialisation) ────────────
    stage          = "SEARCH_BALLOON"
    touch_time     = None      # set when balloon is touched
    classify_start = None      # set when classification phase starts
    votes          = []
    conf_sum       = 0.0
    class_label    = None      # set after classification
    target_tag_id  = None      # set after classification

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

            # ── Update Kalman prediction (always runs) ───────────────────
            dt = now - last_time
            last_time = now
            kf.transitionMatrix[0, 3] = dt
            kf.transitionMatrix[1, 4] = dt
            kf.transitionMatrix[2, 5] = dt
            prediction = kf.predict()

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
                    pid_states     = (0, 0, 0, 0, 0, 0)
                    stage = "TRACK_AND_TOUCH"
                else:
                    Balloon_Detector.search_balloon_pattern(tello, search_speed=25)

            # ── TRACK_AND_TOUCH ──────────────────────────────────────────
            elif stage == "TRACK_AND_TOUCH":
                cv2.putText(display, "Stage 3: Tracking balloon...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                box = Balloon_Detector.detect_balloon(frame, balloon_model)
                tracked_pos = None

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

                    is_touched, pid_states = Balloon_Detector.track_and_control_tello(
                        tello, tracked_pos, pid_states
                    )
                    if is_touched:
                        print("【CP2 待評分】碰撞氣球！TA please judge touch.")
                        tello.send_rc_control(0, 0, 0, 0)
                        touch_time = time.time()
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
                    # No model available — default to 'cap' and warn
                    votes.append('cap')

                cv2.putText(display,
                            f"Stage 4 classifying... {len(votes)}/{CLASSIFY_FRAMES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                # Decide once we have enough votes or hit timeout
                enough = len(votes) >= CLASSIFY_FRAMES or elapsed > CLASSIFY_TIMEOUT
                if enough and votes:
                    counts   = Counter(votes)
                    best, cnt = counts.most_common(1)[0]
                    frac     = cnt / len(votes)
                    avg_conf = conf_sum / max(len(votes), 1)

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
                break   # exit while loop → run approach_and_land below

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
            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                print("[Emergency] ESC pressed → emergency land!")
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                tello.streamoff()
                cv2.destroyAllWindows()
                return

            time.sleep(0.02)   # ~20 Hz frame rate

    except KeyboardInterrupt:
        print("\n[Main] Ctrl-C → landing")
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()
        return

    # ── Stage 4 approach + land ───────────────────────────────────────────
    try:
        approach_and_land(tello, target_tag_id, frame_read)
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