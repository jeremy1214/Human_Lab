"""
Balloon_Detector.py
====================
Stage 2/3 helper functions: balloon detection, Kalman filter, PID tracking.
Used by Final_Game.py — not meant to be run directly.

Detection backend: Ultralytics YOLO (.pt weights).
  pip install ultralytics
No ONNX conversion needed — load balloon.pt directly.
"""

import time

import cv2
import numpy as np

# =====================================================================
# 1. Camera intrinsics & PID gains (Lab 3)
# =====================================================================
FOCAL_LENGTH_X = 920.0        # Tello camera fx (lab-measured, fine-tune as needed)
FOCAL_LENGTH_Y = 920.0        # Tello camera fy
BALLOON_REAL_DIAMETER = 25.0  # cm — update with the size given on competition day

# Tracking PID gains [Kp, Ki, Kd]
PID_X = [0.4, 0.0, 0.1]       # left/right error → Tello yaw
PID_Y = [0.4, 0.0, 0.1]       # up/down error → Tello throttle
PID_Z = [0.5, 0.0, 0.1]       # forward/back distance → Tello pitch

BALLOON_CONF_THRESH = 0.7     # YOLO confidence threshold for balloon detection


# =====================================================================
# 2. Balloon detection (Ultralytics .pt model)
# =====================================================================

def detect_balloon(frame: np.ndarray, model, conf_thresh: float = BALLOON_CONF_THRESH):
    """
    Detect the balloon in a BGR frame using an Ultralytics YOLO .pt model.

    Parameters
    ----------
    frame : np.ndarray
        BGR camera frame.
    model : ultralytics.YOLO | None
        Loaded balloon YOLO model. If None, returns None immediately.
    conf_thresh : float
        Minimum confidence to accept a detection.

    Returns
    -------
    (x, y, w, h) tuple of the highest-confidence box, or None.
    """
    if model is None:
        return None

    results = model.predict(frame, verbose=False, conf=conf_thresh)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    best  = int(np.argmax(confs))
    if confs[best] < conf_thresh:
        return None

    x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


# =====================================================================
# 3. Kalman Filter (constant-velocity model, Lab 3)
# =====================================================================

def init_kalman_filter():
    """Initialise a 6-state (pos+vel) x 3-measurement (pos) Kalman filter."""
    kf = cv2.KalmanFilter(6, 3, 0)
    kf.transitionMatrix = np.eye(6, dtype=np.float32)   # dt filled in per-frame

    kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1
    kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = 1

    kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(6, dtype=np.float32)
    return kf


def recover_3d_position(bbox, img_w, img_h):
    """Back-project a 2D bounding box to a 3D relative position (cm), via similar triangles."""
    bx, by, bw, bh = bbox
    cx = bx + bw // 2
    cy = by + bh // 2

    u_err = cx - (img_w // 2)
    v_err = (img_h // 2) - cy   # flip so "up" is positive

    z_distance = (BALLOON_REAL_DIAMETER * FOCAL_LENGTH_X) / (float(bw) + 1e-5)
    x_distance = (u_err * z_distance) / FOCAL_LENGTH_X
    y_distance = (v_err * z_distance) / FOCAL_LENGTH_Y

    return np.array([[x_distance], [y_distance], [z_distance]], dtype=np.float32)


# =====================================================================
# 4. PID tracking & touch detection
# =====================================================================

def run_pid_core(error, prev_error, integral, pid_gains):
    """Generic PID core."""
    kp, ki, kd = pid_gains
    integral  += error
    derivative = error - prev_error
    output = (kp * error) + (ki * integral) + (kd * derivative)
    return int(np.clip(output, -100, 100)), error, integral


def track_and_control_tello(tello, tracked_pos, pid_states):
    """
    Send 3-axis PID velocity commands based on the Kalman-filtered 3D position.
    Triggers a forward sprint + touch when close enough and centred.

    Returns (is_touched: bool, updated_pid_states).
    """
    x_p, y_p, z_p = tracked_pos[0][0], tracked_pos[1][0], tracked_pos[2][0]
    err_x_p, int_x, err_y_p, int_y, err_z_p, int_z = pid_states

    yaw_speed, err_x_p, int_x = run_pid_core(x_p, err_x_p, int_x, PID_X)
    ud_speed,  err_y_p, int_y = run_pid_core(y_p, err_y_p, int_y, PID_Y)
    fb_speed,  err_z_p, int_z = run_pid_core(z_p - 35, err_z_p, int_z, PID_Z)

    updated_pid_states = (err_x_p, int_x, err_y_p, int_y, err_z_p, int_z)

    if z_p <= 45.0 and abs(x_p) < 10:
        print("[Action] 進入終點線！執行最後向前衝刺碰撞！")
        tello.send_rc_control(0, 45, 0, 0)
        time.sleep(0.8)
        tello.send_rc_control(0, 0, 0, 0)
        return True, updated_pid_states

    tello.send_rc_control(
        0,
        int(np.clip(fb_speed,  -40, 40)),
        int(np.clip(ud_speed,  -30, 30)),
        int(np.clip(yaw_speed, -30, 30)),
    )
    return False, updated_pid_states


def search_balloon_pattern(tello, search_speed=25):
    """Autonomous yaw-only search rotation (rule-compliant: no manual control)."""
    tello.send_rc_control(0, 0, 0, search_speed)