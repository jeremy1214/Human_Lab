#!/usr/bin/env python3
"""
main.py  —  Fully Autonomous Tello Competition Pipeline
========================================================
Run:  python main.py

No keyboard input is needed during flight.
Press ESC in the camera window for emergency land only.

Pipeline (automatic)
--------------------
  TAKEOFF → SEARCH_BALLOON → TRACK_BALLOON
  → TOUCH_CONFIRM → WAIT_FOR_IMAGE → CLASSIFY
  → NAVIGATE_TAG → APPROACH_TAG → LAND

Quick-tune constants are all at the top of this file.
"""

import enum
import math
import os
import sys
import threading
import time
import warnings

import cv2
import numpy as np
from djitellopy import Tello

from apriltag_detector import AprilTagDetector
# Local modules (same directory)
from ekf_localization import EKFLocalization
from image_classifier import (APPROACH_DISTANCE, CLASS_NAMES, LAND_LAT_TOL,
                              LAND_POS_TOL, LANDING_TAG, ImageClassifier,
                              postprocess_yolo, preprocess_yolo)

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  TUNE THESE FOR YOUR SETUP
# ═══════════════════════════════════════════════════════════════════════════════

TAKEOFF_HEIGHT_CM   = 100   # target hover height after takeoff (cm)
SEARCH_YAW_PCT      = 25    # yaw speed while scanning for balloon (%)
SEARCH_TIMEOUT_S    = 20    # seconds per rotation before widening search

# Balloon tracking PID (output = RC %, -100..100)
BALLOON_YAW_KP      = 55    # proportional gain on horizontal pixel error
BALLOON_UD_KP       = 45    # proportional gain on vertical pixel error
BALLOON_FB_KP       = 60    # proportional gain on area error
BALLOON_FB_BASE     = 15    # base forward speed while approaching (%)
TOUCH_AREA_RATIO    = 0.10  # balloon fills >10% of frame → declare touch

# After touch: hover this long before starting image classification (seconds)
TOUCH_HOVER_S       = 2.0

# Image classification: collect this many frames; majority must agree
CLASSIFY_FRAMES     = 20
MIN_VOTE_FRACTION   = 0.50
CLASSIFY_TIMEOUT_S  = 30    # give up after this long (hover and try again)

# Stage-4 approach PID gains (same as image_classifier.py but referenced here)
STAGE4_FWD_KP       = 0.4
STAGE4_LAT_KP       = 0.5
STAGE4_ALT_KP       = 0.4
STAGE4_YAW_KP       = 0.5

# Landing zone AprilTag IDs and balloon-detect ONNX model
BALLOON_MODEL_PATH  = 'balloon_detect.onnx'  # put your balloon YOLO model here
BRAINROT_MODEL_PATH = 'brainrot_detect.onnx'
MAP_YAML            = 'Apriltag/map/apriltag_map.yaml'

# ═══════════════════════════════════════════════════════════════════════════════

class AutoStage(enum.IntEnum):
    TAKEOFF        = 0
    SEARCH_BALLOON = 1
    TRACK_BALLOON  = 2
    TOUCH_CONFIRM  = 3
    WAIT_FOR_IMAGE = 4
    CLASSIFY       = 5
    NAVIGATE_TAG   = 6
    APPROACH_TAG   = 7
    LAND           = 8
    DONE           = 9


# ─── Simple PID (proportional-only for clean stage transitions) ───────────────

class _PID:
    def __init__(self, kp, ki=0.0, kd=0.0, limit=100.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self._i = 0.0; self._pe = 0.0

    def __call__(self, err, dt=0.05):
        self._i  += err * dt
        d         = (err - self._pe) / max(dt, 1e-6)
        self._pe  = err
        return float(np.clip(self.kp*err + self.ki*self._i + self.kd*d,
                             -self.limit, self.limit))

    def reset(self):
        self._i = 0.0; self._pe = 0.0


# ─── Balloon Detector ─────────────────────────────────────────────────────────

class BalloonDetector:
    """
    Detects the competition balloon in a BGR frame.

    Priority order:
      1. User's balloon YOLO ONNX model (balloon_detect.onnx)
      2. HSV adaptive color blob fallback

    Returns
    -------
    dict  {'cx': int, 'cy': int, 'area_ratio': float}
    or None if not detected.
    """

    def __init__(self, model_path: str = BALLOON_MODEL_PATH):
        self._session    = None
        self._input_name = None

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self._session    = ort.InferenceSession(
                    model_path, providers=['CPUExecutionProvider'])
                self._input_name = self._session.get_inputs()[0].name
                print(f"[BalloonDetector] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[BalloonDetector] YOLO load failed ({e}), using color fallback")
        else:
            print(f"[BalloonDetector] '{model_path}' not found — using color fallback")

    def detect(self, frame: np.ndarray):
        if self._session is not None:
            return self._detect_yolo(frame)
        return self._detect_color(frame)

    # ── YOLO path ─────────────────────────────────────────────────────────────
    def _detect_yolo(self, frame):
        blob = preprocess_yolo(frame)
        out  = self._session.run(None, {self._input_name: blob})
        # Reuse YOLOv8 postprocessor with class 0 = balloon
        rows = out[0][0].T                     # [8400, 5+] or [8400, 4+nclass]
        if rows.shape[1] < 5:
            return None
        # Sum all class scores as object confidence
        conf  = rows[:, 4:].max(axis=1)
        best  = int(np.argmax(conf))
        if conf[best] < 0.15:
            return None
        xc, yc, w, h = rows[best, :4]
        H, W = frame.shape[:2]
        return {
            'cx':         int(xc / 640 * W),
            'cy':         int(yc / 640 * H),
            'area_ratio': float((w * h) / (640 * 640)),
        }

    # ── Color path (adaptive HSV) ─────────────────────────────────────────────
    def _detect_color(self, frame):
        """
        Adaptive balloon finder: detects the largest round blob
        that is NOT near-grey/black/white (i.e., is clearly coloured).
        Works for any vivid balloon colour.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = frame.shape[:2]

        # Mask: high saturation (vivid colour), reasonable value
        s_mask = cv2.inRange(hsv, (0, 80, 60), (180, 255, 255))

        # Morphological clean-up
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(s_mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 400:                          # too small → noise
            return None

        x, y, bw, bh = cv2.boundingRect(best)
        # Circularity check — balloon is roughly round
        circ = 4 * math.pi * area / (cv2.arcLength(best, True) ** 2 + 1e-6)
        if circ < 0.35:
            return None

        return {
            'cx':         x + bw // 2,
            'cy':         y + bh // 2,
            'area_ratio': float(area / (H * W)),
        }


# ─── Autonomous Pilot (full state machine) ────────────────────────────────────

class AutonomousPilot:
    """
    Controls the Tello through all competition stages without human input.

    Parameters
    ----------
    tello   : djitellopy.Tello (connected + streaming)
    ekf     : EKFLocalization
    tag_det : AprilTagDetector  (for EKF localisation, tags 4-12)
    """

    def __init__(self, tello: Tello, ekf: EKFLocalization,
                 tag_det: AprilTagDetector):
        self.tello   = tello
        self.ekf     = ekf
        self.tag_det = tag_det

        self.balloon_det = BalloonDetector(BALLOON_MODEL_PATH)

        # Load brainrot classifier
        self._brainrot_session    = None
        self._brainrot_input_name = None
        if os.path.exists(BRAINROT_MODEL_PATH):
            import onnxruntime as ort
            sess = ort.InferenceSession(BRAINROT_MODEL_PATH,
                                        providers=['CPUExecutionProvider'])
            self._brainrot_session    = sess
            self._brainrot_input_name = sess.get_inputs()[0].name
            print(f"[Pilot] Brainrot model loaded.")
        else:
            print(f"[Pilot] WARNING: {BRAINROT_MODEL_PATH} not found!")

        # AprilTag detector for landing-zone tags (13-16, real-time)
        from pupil_apriltags import Detector as _ATDet
        self._at_detector = _ATDet(families='tag36h11', nthreads=1,
                                    quad_decimate=1.0, quad_sigma=0.0,
                                    refine_edges=1, decode_sharpening=0.25)
        self._cam_params = [835.342103847164, 839.4691450667409,
                            415.5366635247159, 355.11975613817964]
        self._tag_size   = 0.165

        # ── Stage machine state ───────────────────────────────────────────────
        self.stage        = AutoStage.TAKEOFF
        self._stage_entry = time.time()

        # CLASSIFY
        self._votes: list = []
        self._class_label = None
        self._target_tag  = None

        # APPROACH
        self._tag_in_view = None
        self._pid_fwd = _PID(STAGE4_FWD_KP, limit=25)
        self._pid_lat = _PID(STAGE4_LAT_KP, limit=20)
        self._pid_alt = _PID(STAGE4_ALT_KP, limit=20)
        self._pid_yaw = _PID(STAGE4_YAW_KP, limit=35)

        # TOUCH_CONFIRM
        self._touch_t = None

        # DISPLAY (annotated frame for monitoring)
        self.display_frame: np.ndarray = None

    # ─── Called every frame (~20 Hz) ────────────────────────────────────────

    def step(self, frame: np.ndarray, dt: float):
        """Run one cycle of the state machine and send RC to Tello."""
        now = time.time()

        # Always try to update EKF from localisation tags 4-12
        pose = self.tag_det.detect(frame)
        if pose is not None:
            self.ekf.update_from_detection(pose)

        # Annotate frame for display
        display = frame.copy()
        self._draw_overlay(display)

        # Stage dispatch
        if   self.stage == AutoStage.TAKEOFF:        self._do_takeoff(dt)
        elif self.stage == AutoStage.SEARCH_BALLOON:  self._do_search(frame, dt)
        elif self.stage == AutoStage.TRACK_BALLOON:   self._do_track(frame, dt)
        elif self.stage == AutoStage.TOUCH_CONFIRM:   self._do_touch_confirm(dt)
        elif self.stage == AutoStage.WAIT_FOR_IMAGE:  self._do_wait_image(dt)
        elif self.stage == AutoStage.CLASSIFY:        self._do_classify(frame, dt)
        elif self.stage == AutoStage.NAVIGATE_TAG:    self._do_navigate(frame, dt)
        elif self.stage == AutoStage.APPROACH_TAG:    self._do_approach(frame, dt)
        elif self.stage == AutoStage.LAND:            self._do_land()
        elif self.stage == AutoStage.DONE:            self._rc(0,0,0,0)

        self.display_frame = display

    # ─── Stage implementations ───────────────────────────────────────────────

    def _do_takeoff(self, dt):
        """Take off and climb to TAKEOFF_HEIGHT_CM."""
        if self._stage_time() < 0.2:
            print("[Pilot] Taking off...")
            self.tello.takeoff()
            time.sleep(3.0)           # let takeoff settle
            print("[Pilot] Airborne — climbing to search height")

        h = self._get_height()
        if h >= TAKEOFF_HEIGHT_CM - 10:
            self._transition(AutoStage.SEARCH_BALLOON)
            return

        # Climb
        self._rc(0, 0, 30, 0)

    def _do_search(self, frame, dt):
        """Rotate slowly until balloon is detected."""
        det = self.balloon_det.detect(frame)
        if det is not None:
            print(f"[Pilot] Balloon found (area={det['area_ratio']:.3f})")
            self._transition(AutoStage.TRACK_BALLOON)
            return

        # Widen search height every SEARCH_TIMEOUT_S seconds
        elapsed = self._stage_time()
        if elapsed > 0 and elapsed % SEARCH_TIMEOUT_S < dt:
            h = self._get_height()
            new_ud = 20 if h < 150 else -20
            print(f"[Pilot] No balloon after {elapsed:.0f}s — adjusting height")
            self._rc(0, 0, new_ud, 0)
            time.sleep(0.5)

        self._rc(0, 0, 0, SEARCH_YAW_PCT)    # keep rotating

    def _do_track(self, frame, dt):
        """Track balloon and approach until touch."""
        det = self.balloon_det.detect(frame)
        H, W = frame.shape[:2]

        if det is None:
            # Lost balloon — spin slowly to re-acquire
            self._rc(0, 0, 0, SEARCH_YAW_PCT)
            if self._stage_time() > 3.0:
                print("[Pilot] Balloon lost — returning to SEARCH")
                self._transition(AutoStage.SEARCH_BALLOON)
            return

        # Reset lost-balloon timer when detected
        self._stage_entry = time.time()  # keep resetting so timeout doesn't fire

        cx, cy  = det['cx'], det['cy']
        area    = det['area_ratio']

        # Touch condition: balloon fills enough of the frame
        if area >= TOUCH_AREA_RATIO:
            print(f"[Pilot] TOUCH detected (area={area:.3f}) — hovering")
            self._rc(0, 0, 0, 0)
            self._touch_t = time.time()
            self._transition(AutoStage.TOUCH_CONFIRM)
            return

        # Errors relative to image centre (normalised -0.5 … +0.5)
        err_yaw = (cx - W / 2) / W        # positive → balloon right  → yaw right
        err_ud  = (cy - H / 2) / H        # positive → balloon below  → go up (negate)
        err_fb  = TOUCH_AREA_RATIO - area  # positive → too far        → go forward

        yaw = int(np.clip( BALLOON_YAW_KP * err_yaw,  -50, 50))
        ud  = int(np.clip(-BALLOON_UD_KP  * err_ud,   -30, 30))
        fb  = int(np.clip( BALLOON_FB_KP  * err_fb + BALLOON_FB_BASE, 0, 50))
        lr  = 0   # don't strafe — yaw to centre instead

        self._rc(lr, fb, ud, yaw)

    def _do_touch_confirm(self, dt):
        """Hover briefly after touching balloon (rule T-R3)."""
        self._rc(0, 0, 0, 0)
        if time.time() - self._touch_t >= TOUCH_HOVER_S:
            print("[Pilot] Touch confirmed — waiting for TA to show image")
            self._votes = []
            self._transition(AutoStage.WAIT_FOR_IMAGE)

    def _do_wait_image(self, dt):
        """
        Hover and wait.
        Transition to CLASSIFY once we've had a moment to stabilise,
        giving the TA time to walk up and show the image.
        """
        self._rc(0, 0, 0, 0)
        # Wait 3 seconds for the TA to position the image
        if self._stage_time() >= 3.0:
            print("[Pilot] Starting image classification")
            self._votes = []
            self._transition(AutoStage.CLASSIFY)

    def _do_classify(self, frame, dt):
        """Collect YOLO votes until confident, then go to navigation."""
        self._rc(0, 0, 0, 0)    # hover

        if self._brainrot_session is None:
            print("[Pilot] No brainrot model — cannot classify!")
            return

        # Run inference
        blob = preprocess_yolo(frame)
        out  = self._brainrot_session.run(
                   None, {self._brainrot_input_name: blob})
        dets = postprocess_yolo(out[0])
        if dets:
            self._votes.append(dets[0]['class_name'])

        # Decide
        if len(self._votes) >= CLASSIFY_FRAMES:
            counts = {c: self._votes.count(c) for c in CLASS_NAMES}
            best   = max(counts, key=counts.get)
            frac   = counts[best] / len(self._votes)

            if frac >= MIN_VOTE_FRACTION:
                self._class_label = best
                self._target_tag  = LANDING_TAG[best]
                print(f"\n{'='*40}")
                print(f"  CLASSIFICATION : {best.upper()}")
                print(f"  TARGET TAG     : {self._target_tag}")
                print(f"{'='*40}\n")
                for pid in (self._pid_fwd, self._pid_lat,
                            self._pid_alt, self._pid_yaw):
                    pid.reset()
                self._transition(AutoStage.NAVIGATE_TAG)
            else:
                print(f"[Pilot] Low agreement ({frac:.0%}) — collecting more frames")
                self._votes = []

        # Timeout: if classification taking too long, retry
        if self._stage_time() > CLASSIFY_TIMEOUT_S:
            print("[Pilot] Classification timeout — resetting vote buffer")
            self._votes = []
            self._stage_entry = time.time()

    def _do_navigate(self, frame, dt):
        """Rotate until target landing-zone AprilTag is visible."""
        tag = self._find_landing_tag(frame)
        if tag is not None:
            self._tag_in_view = tag
            print(f"[Pilot] Tag {self._target_tag} acquired — approaching")
            for pid in (self._pid_fwd, self._pid_lat,
                        self._pid_alt, self._pid_yaw):
                pid.reset()
            self._transition(AutoStage.APPROACH_TAG)
            return
        self._rc(0, 0, 0, SEARCH_YAW_PCT)

    def _do_approach(self, frame, dt):
        """PID approach to land 40-70 cm in front of the target tag."""
        tag = self._find_landing_tag(frame)

        if tag is None:
            print("[Pilot] Tag lost — returning to NAVIGATE")
            self._rc(0, 0, 0, 0)
            self._transition(AutoStage.NAVIGATE_TAG)
            return

        self._tag_in_view = tag
        tx = float(tag.pose_t[0])
        ty = float(tag.pose_t[1])
        tz = float(tag.pose_t[2])

        err_fwd = tz - APPROACH_DISTANCE
        err_lat = tx
        err_alt = ty
        yaw_err = math.atan2(tag.pose_R[1, 0], tag.pose_R[0, 0])

        fb  = int(np.clip( self._pid_fwd(err_fwd, dt) * 100, -25, 25))
        lr  = int(np.clip(-self._pid_lat(err_lat, dt) * 100, -20, 20))
        ud  = int(np.clip(-self._pid_alt(err_alt, dt) * 100, -20, 20))
        yaw = int(np.clip(-self._pid_yaw(yaw_err, dt) * 100, -35, 35))
        self._rc(lr, fb, ud, yaw)

        if (abs(err_fwd) < LAND_POS_TOL and
                abs(err_lat) < LAND_LAT_TOL and
                abs(err_alt) < LAND_LAT_TOL):
            print("[Pilot] In landing position")
            self._rc(0, 0, 0, 0)
            self._transition(AutoStage.LAND)

    def _do_land(self):
        self._rc(0, 0, 0, 0)
        print("[Pilot] Landing...")
        self.tello.land()
        self._transition(AutoStage.DONE)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _rc(self, lr, fb, ud, yaw):
        self.tello.send_rc_control(
            int(np.clip(lr,  -100, 100)),
            int(np.clip(fb,  -100, 100)),
            int(np.clip(ud,  -100, 100)),
            int(np.clip(yaw, -100, 100)),
        )

    def _transition(self, new_stage: AutoStage):
        print(f"[Pilot] {self.stage.name} → {new_stage.name}")
        self.stage        = new_stage
        self._stage_entry = time.time()

    def _stage_time(self) -> float:
        return time.time() - self._stage_entry

    def _get_height(self) -> int:
        try:
            return self.tello.get_height()
        except Exception:
            return 0

    def _find_landing_tag(self, frame):
        if self._target_tag is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self._at_detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=self._cam_params, tag_size=self._tag_size)
        for t in tags:
            if t.tag_id == self._target_tag:
                return t
        return None

    # ─── Display overlay ─────────────────────────────────────────────────────

    def _draw_overlay(self, frame):
        H, W = frame.shape[:2]
        BAR_H = 120
        bar   = np.zeros((BAR_H, W, 3), dtype=np.uint8)

        stage_color = {
            AutoStage.TAKEOFF:        (180, 180,  40),
            AutoStage.SEARCH_BALLOON: (255, 165,   0),
            AutoStage.TRACK_BALLOON:  (  0, 200, 255),
            AutoStage.TOUCH_CONFIRM:  (  0, 200,   0),
            AutoStage.WAIT_FOR_IMAGE: (180, 180,  40),
            AutoStage.CLASSIFY:       (255, 165,   0),
            AutoStage.NAVIGATE_TAG:   (  0, 200, 255),
            AutoStage.APPROACH_TAG:   (  0, 200,   0),
            AutoStage.LAND:           ( 50,  50, 220),
            AutoStage.DONE:           (150, 150, 150),
        }.get(self.stage, (200, 200, 200))

        def txt(text, x, y, color=(220,220,220), scale=0.6, thick=1):
            cv2.putText(bar, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                        cv2.LINE_AA)

        txt(f"STAGE: {self.stage.name}", 10, 28, stage_color, 0.8, 2)
        txt(f"t={self._stage_time():.1f}s", 380, 28)
        try:
            bat = self.tello.get_battery()
            h   = self.tello.get_height()
            txt(f"Battery:{bat}%  Height:{h}cm", 10, 55)
        except Exception:
            txt("Battery:---  Height:---", 10, 55)

        if self._class_label:
            txt(f"Class:{self._class_label}  Tag:{self._target_tag}",
                10, 80, (0,255,120))

        if self.ekf.is_initialized:
            p = self.ekf.pose
            txt(f"EKF x={p[0]:.2f} y={p[1]:.2f} z={p[2]:.2f}  "
                f"yaw={math.degrees(p[4]):.1f}d", 10, 105, (100,220,255))

        txt("ESC = emergency land", 10, BAR_H - 6, (100,100,100), 0.45)

        combined = np.vstack([bar, frame])
        frame[:] = combined[:H]     # write back into original array


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND THREADS
# ═══════════════════════════════════════════════════════════════════════════════

def _state_thread(tello: Tello, ekf: EKFLocalization):
    """Poll Tello state at 10 Hz and step the EKF prediction."""
    last_roll = last_pitch = last_yaw = 0.0
    last_t    = time.time()
    while True:
        now = time.time()
        dt  = now - last_t;  last_t = now
        try:
            vx  =  float(tello.get_state_field('vgx')) / 100.0
            vy  = -float(tello.get_state_field('vgy')) / 100.0
            vz  = -float(tello.get_state_field('vgz')) / 100.0
            rr  = math.radians(tello.get_roll())
            pr  = math.radians(tello.get_pitch())
            yr  = math.radians(tello.get_yaw())

            def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi
            if dt > 0.001:
                rrate = wrap(rr - last_roll)  / dt
                prate = wrap(pr - last_pitch) / dt
                yrate = wrap(yr - last_yaw)   / dt
            else:
                rrate = prate = yrate = 0.0
            last_roll, last_pitch, last_yaw = rr, pr, yr

            ekf.set_control(vx, vy, vz, rrate, yrate, prate)
            ekf.step(dt)
        except Exception:
            pass
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 52)
    print("  Tello Autonomous Competition Pipeline")
    print("  Connect to Tello Wi-Fi, then press Enter")
    print("=" * 52)
    input("  >> Press Enter to start <<")

    # ── Connect ──────────────────────────────────────────────────────────────
    tello = Tello()
    tello.connect()
    print(f"[main] Connected.  Battery: {tello.get_battery()} %")
    if tello.get_battery() < 20:
        print("[main] WARNING: battery low!")
    tello.streamon()
    frame_read = tello.get_frame_read()
    time.sleep(1.0)

    # ── EKF + AprilTag localisation ──────────────────────────────────────────
    ekf     = EKFLocalization()
    tag_det = AprilTagDetector(MAP_YAML)

    # ── Background state thread ───────────────────────────────────────────────
    st = threading.Thread(target=_state_thread, args=(tello, ekf), daemon=True)
    st.start()

    # ── Autonomous pilot ──────────────────────────────────────────────────────
    pilot   = AutonomousPilot(tello, ekf, tag_det)
    last_t  = time.time()

    print("[main] Starting autonomous flight in 3 seconds...")
    time.sleep(3.0)

    try:
        while pilot.stage != AutoStage.DONE:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.02)
                continue

            now  = time.time()
            dt   = now - last_t
            last_t = now

            pilot.step(frame, dt)

            # ── Display ──────────────────────────────────────────────────────
            show = pilot.display_frame if pilot.display_frame is not None else frame
            cv2.imshow('Tello Autonomous', show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                print("[main] ESC pressed — emergency land!")
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                break

            time.sleep(max(0, 0.05 - (time.time() - now)))  # ~20 Hz

    except KeyboardInterrupt:
        print("[main] Interrupted")

    finally:
        print("[main] Shutting down...")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[main] Done.")


if __name__ == '__main__':
    main()