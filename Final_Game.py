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
  [Stage 4] Brainrot YOLO classification (majority vote)
  [Stage 4] Rotate until landing AprilTag visible
  [Stage 4] PID approach → land

Bugs fixed vs original Final_Game.py
-------------------------------------
  1. postprocess_yolo: confidence was confs_f[idx_c.index(i)] → now cfs_c[k]
  2. subprocess ROS node launch removed; classification runs inline
  3. drawn_angle random.choice replaced with pre-flight user input
  4. Dead-reckoning center-flight replaced with AprilTag PID approach
  5. touch_time / classify_start / class_label scope made explicit
  6. EKF inv(5) bug noted (in ekf_localization_node.py — fixed in that file)
"""

import math
import os
import random
import time
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort
from pupil_apriltags import Detector as ATDetector

import Balloon_Detector  # Stage 2 / 3 logic (Kalman + PID + touch)
import Start_Tello  # Stage 1 takeoff helpers

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  tune these numbers before the competition
# ═══════════════════════════════════════════════════════════════════════════════

BALLOON_ONNX    = 'balloon.onnx'          # balloon YOLOv5 model (onnxruntime)
BRAINROT_ONNX   = 'brainrot_detect.onnx'  # brainrot YOLOv8 model (onnxruntime)

# ── AprilTag camera intrinsics (Lab 1 calibration) ──────────────────────────
AT_FX, AT_FY    = 835.342103847164, 839.4691450667409
AT_CX, AT_CY    = 415.5366635247159, 355.11975613817964
AT_TAG_SIZE     = 0.165   # metres — update if landing tags differ

# ── Classification ────────────────────────────────────────────────────────────
CLASS_NAMES       = ['cap', 'brr', 'trala', 'tung']
LANDING_TAG       = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}
CLASSIFY_FRAMES   = 20    # vote-collection window
MIN_VOTE_FRACTION = 0.50  # majority needed
CLASSIFY_TIMEOUT  = 30.0  # seconds; reset & retry if inconclusive

# ── Stage-4 approach ─────────────────────────────────────────────────────────
APPROACH_DIST_M = 0.55    # metres in front of tag (40–70 cm spec)
LAND_POS_TOL    = 0.10    # forward-error tolerance (m) to trigger land
LAND_LAT_TOL    = 0.08    # lateral / vertical tolerance (m)
SEARCH_YAW_PCT  = 25      # RC % while rotating to find landing tag

# Proportional gains (output in RC %, clamped to limit)
KP_FWD, KP_LAT, KP_ALT, KP_YAW = 40.0, 50.0, 40.0, 50.0

# ── Timing ────────────────────────────────────────────────────────────────────
POST_TOUCH_HOVER_S = 3.0  # hover after touch so TA can walk up with image

# ── YOLO (brainrot) ───────────────────────────────────────────────────────────
YOLO_INPUT_SIZE  = 640
YOLO_CONF_THRESH = 0.15
YOLO_NMS_THRESH  = 0.45


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAINROT YOLO v8 HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh))
    py, px = (size - nh) // 2, (size - nw) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def preprocess_yolo(frame):
    lb   = _letterbox(frame, YOLO_INPUT_SIZE)
    rgb  = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0)   # [1,3,640,640]
    return blob


def postprocess_yolo(output):
    """
    Parse YOLOv8 ONNX output [1, 8, 8400].
    Returns list of {'class_name', 'confidence'} sorted by confidence desc.

    Bug fix: original code incorrectly re-indexed the full confidence array.
    Correct form: cfs_c[k], where cfs_c = confs_f[idx_c].
    """
    rows        = output[0].T                          # [8400, 8]
    class_ids   = np.argmax(rows[:, 4:], axis=1)
    confidences = rows[np.arange(len(class_ids)), 4 + class_ids]
    mask        = confidences >= YOLO_CONF_THRESH
    if not np.any(mask):
        return []

    boxes_f = rows[mask, :4]
    confs_f = confidences[mask]
    class_f = class_ids[mask]

    x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
    y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
    x2 = boxes_f[:, 0] + boxes_f[:, 2] / 2
    y2 = boxes_f[:, 1] + boxes_f[:, 3] / 2

    results = []
    for cid in np.unique(class_f):
        # Skip classes that are not present in our CLASS_NAMES mapping
        if int(cid) < 0 or int(cid) >= len(CLASS_NAMES):
            # unexpected class id from model — skip to avoid index errors
            continue
        idx_c  = np.where(class_f == cid)[0]
        bxs_c  = np.stack([x1[idx_c], y1[idx_c],
                            x2[idx_c] - x1[idx_c],
                            y2[idx_c] - y1[idx_c]], axis=1).tolist()
        cfs_c  = confs_f[idx_c].tolist()   # confidences for this class only
        keep   = cv2.dnn.NMSBoxes(bxs_c, cfs_c, YOLO_CONF_THRESH, YOLO_NMS_THRESH)
        if len(keep) == 0:
            continue
        for k in (keep.flatten() if isinstance(keep, np.ndarray) else keep):
            results.append({
                'class_name': CLASS_NAMES[int(cid)],
                'confidence': float(cfs_c[k]),   # ← bug fix
            })
    return sorted(results, key=lambda d: d['confidence'], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: AprilTag PID approach → land
# ═══════════════════════════════════════════════════════════════════════════════

def _rc_clamp(val, limit):
    return int(np.clip(val, -limit, limit))


def approach_and_land(tello, target_tag_id: int, frame_read):
    """
    Rotate until the target landing AprilTag is visible then PID-approach
    to APPROACH_DIST_M and land.  ESC in the window triggers emergency land.
    """
    try:
        at_det = ATDetector(
            families='tag36h11', nthreads=1,
            quad_decimate=1.0, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.25, debug=0,
        )
    except Exception as e:
        print(f"[Stage 4] ERROR initializing AprilTag detector: {e}")
        print("[Stage 4] Performing emergency land")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception:
            pass
        return
    cam_params = [AT_FX, AT_FY, AT_CX, AT_CY]

    print(f"[Stage 4] Searching for AprilTag id={target_tag_id}...")

    while True:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.02)
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_det.detect(
                gray, estimate_tag_pose=True,
                camera_params=cam_params, tag_size=AT_TAG_SIZE,
            )
        except Exception as e:
            print(f"[Stage 4] AprilTag detection error: {e} — emergency land")
            try:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
            except Exception:
                pass
            return

        target = next((t for t in tags if t.tag_id == target_tag_id), None)
        display = frame.copy()

        if target is None:
            # Spin slowly to scan for the tag
            tello.send_rc_control(0, 0, 0, SEARCH_YAW_PCT)
            cv2.putText(display, f"Stage 4: rotating to find tag {target_tag_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # Camera-frame translation (OpenCV: +X right, +Y down, +Z forward)
            tx = float(target.pose_t[0])
            ty = float(target.pose_t[1])
            tz = float(target.pose_t[2])

            err_fwd = tz - APPROACH_DIST_M   # + → still too far
            err_lat = tx                       # + → tag right → move right
            err_alt = ty                       # + → tag below → go up
            yaw_err = math.atan2(target.pose_R[1, 0], target.pose_R[0, 0])

            fb  = _rc_clamp( KP_FWD * err_fwd,  25)
            lr  = _rc_clamp(-KP_LAT * err_lat,  20)   # negate: right → move right (lr+)
            ud  = _rc_clamp(-KP_ALT * err_alt,  20)   # negate: below → go up (ud+)
            yaw = _rc_clamp(-KP_YAW * yaw_err,  35)

            tello.send_rc_control(lr, fb, ud, yaw)

            cv2.putText(display,
                        f"Stage 4: tag {target_tag_id}  dist={tz:.2f}m  lat={tx:.2f}m  alt={ty:.2f}m",
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

    # ── Pre-flight: wait for start confirmation ─────────────────────────
    input("\n  Connect laptop to Tello Wi-Fi, then press Enter to start Stage 1...")

    # ── Load models ───────────────────────────────────────────────────────
    balloon_sess  = None
    brainrot_sess = None
    brainrot_inp  = None

    if os.path.exists(BALLOON_ONNX):
        try:
            balloon_sess = ort.InferenceSession(BALLOON_ONNX, providers=['CPUExecutionProvider'])
            print(f"[Init] Balloon model loaded with ONNX Runtime ({BALLOON_ONNX})")
        except Exception as exc:
            print(f"[WARNING] Balloon model load failed in ONNX Runtime: {exc}")
            print("[WARNING] Balloon detection will fall back to color-based detection.")
            balloon_sess = None
    else:
        print(f"[WARNING] {BALLOON_ONNX} not found — using color-based balloon detection fallback.")

    if os.path.exists(BRAINROT_ONNX):
        brainrot_sess = ort.InferenceSession(BRAINROT_ONNX,
                                              providers=['CPUExecutionProvider'])
        brainrot_inp  = brainrot_sess.get_inputs()[0].name
        print(f"[Init] Brainrot model loaded ({BRAINROT_ONNX})")
    else:
        print(f"[WARNING] {BRAINROT_ONNX} not found — Stage 4 classification disabled.")

    # ── Stage 1: takeoff + rotate ─────────────────────────────────────────
    tello = Start_Tello.initialize_tello_stage1()
    Start_Tello.rotate_to_start_angle(tello)
    frame_read = tello.get_frame_read()

    # ── Kalman + PID state ────────────────────────────────────────────────
    kf             = Balloon_Detector.init_kalman_filter()
    kf_initialized = False
    lost_counter   = 0
    pid_states     = (0, 0, 0, 0, 0, 0)
    last_time      = time.time()

    # ── Stage state machine (explicit variable initialisation) ────────────
    stage = "TAKEOFF"  # 起始狀態
    # 建立 AprilTag 偵測器實例
    from apriltag_detector import AprilTagDetector
    tag_detector = AprilTagDetector('apriltag_map.yaml')
    touch_time    = None      # set when balloon is touched
    classify_start = None     # set when classification phase starts
    votes         = []
    conf_sum      = 0.0
    class_label   = None      # set after classification
    target_tag_id = None      # set after classification

    print("[FSM] Entering competition loop  —  ESC in camera window = emergency land")

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

            display = frame.copy()
            h, w, _ = frame.shape
            now   = time.time()

            # ── Update Kalman prediction (always runs) ───────────────────
            dt = now - last_time
            last_time = now
            kf.transitionMatrix[0, 3] = dt
            kf.transitionMatrix[1, 4] = dt
            kf.transitionMatrix[2, 5] = dt
            prediction = kf.predict()

            box = Balloon_Detector.detect_balloon(frame, ort_sess=balloon_sess)

            if stage == "TAKEOFF":
                tello.takeoff()
                time.sleep(1.0)
                # 呼叫 Stage 1 的定量旋轉 (drawn_angle 來自你的預先輸入)
                Start_Tello.rotate_to_start_angle(tello)
                
                # ✨ 關鍵修改：旋轉完不直接找氣球，先進入 AprilTag 定位導航狀態
                stage = "GO_TO_APRILTAG"
                print("[FSM] 轉向完畢，切換至 GO_TO_APRILTAG 階段...")

            elif stage == "GO_TO_APRILTAG":
                # 呼叫上面寫好的導航函式，移至地圖中的 Tag 4 前方 1.2 公尺處
                # 你可以根據實際場地將 target_tag_id 設為靠近起飛點的 Landing Tag ID
                is_arrived = Balloon_Detector.navigate_to_reference_tag(tello, tag_detector, frame, 
                                                        target_tag_id=4, 
                                                        hold_distance_m=1.2)
                if is_arrived:
                    print("[FSM] 導航就位！切換至 SEARCH_BALLOON 開始自轉搜尋氣球...")
                    stage = "SEARCH_BALLOON"
            # ── SEARCH_BALLOON ───────────────────────────────────────────
            elif stage == "SEARCH_BALLOON":
                cv2.putText(display, "Stage 2: Searching for balloon...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
                        # stage = "POST_TOUCH_HOVER"

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

                if brainrot_sess is not None:
                    blob = preprocess_yolo(frame)
                    out  = brainrot_sess.run(None, {brainrot_inp: blob})
                    dets = postprocess_yolo(out[0])
                    if dets:
                        votes.append(dets[0]['class_name'])
                        conf_sum += dets[0]['confidence']
                        cv2.putText(display,
                                    f"Detected: {dets[0]['class_name']} "
                                    f"({dets[0]['confidence']:.2f})",
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
                    counts         = Counter(votes)
                    best, cnt      = counts.most_common(1)[0]
                    frac           = cnt / len(votes)
                    avg_conf       = conf_sum / max(len(votes), 1)

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
                        # Low agreement — collect more frames
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
                bat = tello.get_battery()
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