"""
Balloon_Detector.py
====================
Stage 2/3 helper functions: balloon detection, Kalman filter, PID tracking.
Used by Final_Game.py — not meant to be run directly.

Detection backend: Ultralytics YOLO (.pt weights).

Bug fixes vs the previous version (see Final_Game.py for the full writeup):
  1. FOCAL_LENGTH_X/Y were a rough guess (920.0/920.0). Replaced with the
     same Lab-1-calibrated values already trusted for AprilTag detection
     (835.34 / 839.47) — using two different focal lengths for the same
     physical camera was the most likely cause of the wrong distance
     estimate ("balloon still far but thinks it has reached target").
  2. The old run_pid_core() ignored dt entirely (no time normalisation on
     the integral/derivative terms) — gains tuned for one frame rate would
     misbehave at another. Replaced with the shared, dt-aware PIDController
     (see pid_controller.py, written to the same spec as pid.py's PID class).
  3. The touch condition fired on a SINGLE frame where z_p <= threshold,
     with no debounce — one noisy detection (YOLO box briefly too wide)
     could trigger a full-speed sprint while the balloon was still far
     away. BalloonTracker now requires TOUCH_CONFIRM_FRAMES consecutive
     close+centred frames before reporting ready_to_sprint=True, and no
     longer blocks inside update() with time.sleep() — the actual sprint
     is executed by Final_Game.py as its own non-blocking stage so the
     camera feed and event loop keep running during it.
"""

import cv2
import numpy as np

from pid_controller import PIDController

# =====================================================================
# 1. Camera intrinsics & PID gains
# =====================================================================
# Same calibrated values used for AprilTag detection (Lab 1) — using a
# different, uncalibrated focal length here was the root cause of the
# wrong balloon-distance estimate.
FOCAL_LENGTH_X = 835.342103847164
FOCAL_LENGTH_Y = 839.4691450667409

BALLOON_REAL_DIAMETER = 25.0  # cm — update with the size given on competition day

# Tracking PID gains [Kp, Ki, Kd]
PID_X_GAINS = (0.4, 0.0, 0.08)   # left/right error → Tello yaw
PID_Y_GAINS = (0.4, 0.0, 0.08)   # up/down error    → Tello throttle
PID_Z_GAINS = (0.5, 0.0, 0.10)   # forward distance → Tello pitch

BALLOON_CONF_THRESH = 0.75     # YOLO confidence threshold for balloon detection

# Touch / sprint thresholds
TOUCH_STANDOFF_CM    = 35.0   # PID_Z setpoint: hold this far from the balloon
TOUCH_DISTANCE_CM    = 60.0   # once this close AND centred, consider "ready"
TOUCH_CENTER_TOL_CM  = 15.0   # lateral tolerance for "centred"
TOUCH_HEIGHT_TOL_CM  = 10.0
TOUCH_CONFIRM_FRAMES = 4      # require this many CONSECUTIVE ready frames
                              # (debounce against a single noisy detection)


# =====================================================================
# 2. Balloon detection (Ultralytics .pt model)
# =====================================================================

def detect_balloon(frame: np.ndarray, model, conf_thresh: float = BALLOON_CONF_THRESH):
    """
    Detect the balloon in a BGR frame using an Ultralytics YOLO .pt model.

    Returns (x, y, w, h) of the highest-confidence box, or None.
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
# 4. Balloon tracker: PID + debounced touch detection
# =====================================================================

class BalloonTracker:
    """
    Stateful tracking controller. Create ONE instance per attempt (call
    .reset() when re-entering SEARCH_BALLOON after losing the balloon).

    update() never blocks and never sends a sprint by itself — it just
    reports when the drone has been close + centred for TOUCH_CONFIRM_FRAMES
    consecutive frames, via ready_to_sprint. The caller (Final_Game.py) is
    responsible for executing the actual sprint as its own non-blocking
    stage so the camera/event loop keeps running.
    """

    def __init__(self):
        self.pid_yaw = PIDController(*PID_X_GAINS, output_limit=100, integral_limit=40)
        self.pid_ud  = PIDController(*PID_Y_GAINS, output_limit=100, integral_limit=40)
        self.pid_fb  = PIDController(*PID_Z_GAINS, output_limit=100, integral_limit=40)
        self.close_streak = 0

    def reset(self):
        self.pid_yaw.reset()
        self.pid_ud.reset()
        self.pid_fb.reset()
        self.close_streak = 0

    def update(self, tracked_pos, dt: float):
        """
        Parameters
        ----------
        tracked_pos : Kalman-filtered [[x],[y],[z]] (cm), from kf.statePost[0:3]
        dt : float
            Seconds since the last call.

        Returns
        -------
        (lr, fb, ud, yaw, ready_to_sprint)
            lr is always 0 (lateral correction is done via yaw, not strafing).
        """
        x_p = float(tracked_pos[0][0])
        y_p = float(tracked_pos[1][0])
        z_p = float(tracked_pos[2][0])

        yaw_speed = self.pid_yaw.update(x_p, dt)
        ud_speed  = self.pid_ud.update(y_p - TOUCH_HEIGHT_TOL_CM, dt)
        fb_speed  = self.pid_fb.update(z_p - TOUCH_STANDOFF_CM, dt)

        is_close_now = (z_p <= TOUCH_DISTANCE_CM) and (abs(x_p) < TOUCH_CENTER_TOL_CM) and (abs(y_p) < TOUCH_HEIGHT_TOL_CM)
        self.close_streak = self.close_streak + 1 if is_close_now else 0
        ready_to_sprint    = self.close_streak >= TOUCH_CONFIRM_FRAMES

        lr  = 0
        fb  = int(np.clip(fb_speed,  -40, 40))
        ud  = int(np.clip(ud_speed,  -30, 30))
        yaw = int(np.clip(yaw_speed, -30, 30))
        return lr, fb, ud, yaw, ready_to_sprint


def search_balloon_pattern(tello, search_speed=-35):
    """Autonomous yaw-only search rotation (rule-compliant: no manual control)."""
    tello.send_rc_control(0, 0, 0, search_speed)