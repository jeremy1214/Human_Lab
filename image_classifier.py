"""
image_classifier.py
===================
Stage 4: classify a brainrot meme image then navigate to and land
in front of the matching AprilTag landing zone.
No ROS required — uses djitellopy directly.

State machine
-------------
  IDLE → CLASSIFYING → NAVIGATING → APPROACHING → LANDING

Keybindings (pygame window must be focused)
-------------------------------------------
  [C]     Start classification
  [SPACE] Emergency stop → IDLE
  [L]     Force land
  [R]     Reset to IDLE
"""

import enum
import math
import os
import sys
import time

import cv2
import numpy as np
import pygame
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector


# ── Class → landing-zone mapping (competition spec) ──────────────────────────
CLASS_NAMES  = ['cap', 'brr', 'trala', 'tung']
LANDING_TAG  = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}

# ── YOLO inference settings ───────────────────────────────────────────────────
YOLO_INPUT_SIZE  = 640
YOLO_CONF_THRESH = 0.15
YOLO_NMS_THRESH  = 0.45

# ── Vote buffer ───────────────────────────────────────────────────────────────
CLASSIFY_FRAMES   = 15
MIN_VOTE_FRACTION = 0.50

# ── Approach geometry ─────────────────────────────────────────────────────────
APPROACH_DISTANCE = 0.55   # metres from tag face (centre of 40-70 cm range)
LAND_POS_TOL      = 0.10   # metres — all-axes tolerance to trigger land
LAND_LAT_TOL      = 0.08


class Stage(enum.IntEnum):
    IDLE        = 0
    CLASSIFYING = 1
    NAVIGATING  = 2
    APPROACHING = 3
    LANDING     = 4


# ── Simple PID ────────────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self._integral = 0.0
        self._prev_err = 0.0

    def compute(self, error, dt=0.05):
        self._integral += error * dt
        d = (error - self._prev_err) / max(dt, 1e-6)
        self._prev_err = error
        return float(np.clip(self.kp*error + self.ki*self._integral + self.kd*d,
                             -self.limit, self.limit))

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0


# ── YOLO helpers ──────────────────────────────────────────────────────────────

def _letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh))
    py, px = (size-nh)//2, (size-nw)//2
    canvas[py:py+nh, px:px+nw] = resized
    return canvas


def preprocess_yolo(frame):
    lb   = _letterbox(frame, YOLO_INPUT_SIZE)
    rgb  = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(rgb, (2,0,1)), 0)   # [1,3,640,640]
    return blob


def postprocess_yolo(output):
    rows       = output[0].T                    # [8400, 8]
    class_ids  = np.argmax(rows[:, 4:], axis=1)
    confidences= rows[np.arange(len(class_ids)), 4 + class_ids]
    mask       = confidences >= YOLO_CONF_THRESH
    if not np.any(mask):
        return []

    bxs = rows[mask, :4]
    cfs = confidences[mask]
    cls = class_ids[mask]

    x1 = bxs[:,0]-bxs[:,2]/2;  y1 = bxs[:,1]-bxs[:,3]/2
    x2 = bxs[:,0]+bxs[:,2]/2;  y2 = bxs[:,1]+bxs[:,3]/2

    results = []
    for cid in np.unique(cls):
        idx = np.where(cls == cid)[0]
        boxes_cv = [[float(x1[i]), float(y1[i]),
                     float(x2[i]-x1[i]), float(y2[i]-y1[i])] for i in idx]
        keep = cv2.dnn.NMSBoxes(boxes_cv, cfs[idx].tolist(),
                                 YOLO_CONF_THRESH, YOLO_NMS_THRESH)
        if len(keep) == 0:
            continue
        for k in (keep.flatten() if isinstance(keep, np.ndarray) else keep):
            results.append({'class_name': CLASS_NAMES[int(cid)],
                            'confidence': float(cfs[idx[k]])})
    return sorted(results, key=lambda d: d['confidence'], reverse=True)


# ── Main Stage-4 class ────────────────────────────────────────────────────────

class ImageClassifier:
    """
    Parameters
    ----------
    tello : djitellopy.Tello
        Already connected and streaming.
    model_path : str
        Path to brainrot_detect.onnx
    """

    # Camera intrinsics
    FX, FY = 835.342103847164, 839.4691450667409
    CX, CY = 415.5366635247159, 355.11975613817964
    TAG_SIZE = 0.165   # metres — update if landing tags differ

    def __init__(self, tello, model_path: str = 'brainrot_detect.onnx'):
        self.tello = tello

        # ── ONNX model ──────────────────────────────────────────────────────
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session    = ort.InferenceSession(model_path,
                                               providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print(f"[ImageClassifier] Model loaded from {model_path}")

        # ── AprilTag detector for landing-zone tags 13-16 ───────────────────
        self.at_detector = Detector(
            families='tag36h11', nthreads=1,
            quad_decimate=1.0, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.25, debug=0,
        )
        self.camera_params = [self.FX, self.FY, self.CX, self.CY]

        # ── State machine ────────────────────────────────────────────────────
        self.stage         = Stage.IDLE
        self.class_label   = None
        self.target_tag_id = None
        self.tag_in_view   = None   # pupil_apriltags detection
        self.votes: list   = []
        self.conf_sum      = 0.0

        # ── PID controllers ──────────────────────────────────────────────────
        self.pid_fwd = PID(kp=0.4, ki=0.0, kd=0.08, limit=0.25)
        self.pid_lat = PID(kp=0.5, ki=0.0, kd=0.10, limit=0.20)
        self.pid_alt = PID(kp=0.4, ki=0.0, kd=0.08, limit=0.20)
        self.pid_yaw = PID(kp=0.5, ki=0.0, kd=0.05, limit=0.35)

        # ── Pygame window ────────────────────────────────────────────────────
        # NOTE: pygame must already be initialised by the caller
        self.screen = pygame.display.set_mode((520, 360))
        pygame.display.set_caption('Stage 4 — Image Classifier & Landing')
        self.font_l = pygame.font.SysFont('monospace', 22, bold=True)
        self.font_m = pygame.font.SysFont('monospace', 19)
        self.font_s = pygame.font.SysFont('monospace', 15)

        self._running = True

    # ─── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        """Blocking loop.  Run this from the main thread (pygame requirement)."""
        clock = pygame.time.Clock()
        frame_read = self.tello.get_frame_read()

        while self._running:
            frame = frame_read.frame
            if frame is not None:
                self._process_frame(frame)
                self._control_step()

            self._handle_pygame()
            self._render()
            clock.tick(20)

        self._stop()

    # ─── Frame processing ─────────────────────────────────────────────────────

    def _process_frame(self, frame):
        # Always scan for target landing tag
        self.tag_in_view = None
        if self.target_tag_id is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for t in self.at_detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=self.camera_params, tag_size=self.TAG_SIZE
            ):
                if t.tag_id == self.target_tag_id:
                    self.tag_in_view = t
                    break

        # Classification voting
        if self.stage == Stage.CLASSIFYING:
            label, conf = self._classify(frame)
            if label is not None:
                self.votes.append(label)
                self.conf_sum += conf
            if len(self.votes) >= CLASSIFY_FRAMES:
                self._finalise()

    # ─── YOLO inference ───────────────────────────────────────────────────────

    def _classify(self, frame):
        blob = preprocess_yolo(frame)
        out  = self.session.run(None, {self.input_name: blob})
        dets = postprocess_yolo(out[0])
        if not dets:
            return None, 0.0
        return dets[0]['class_name'], dets[0]['confidence']

    def _finalise(self):
        counts    = {c: self.votes.count(c) for c in CLASS_NAMES}
        best      = max(counts, key=counts.get)
        frac      = counts[best] / len(self.votes)
        avg_conf  = self.conf_sum / len(self.votes)

        print(f"\n{'='*40}")
        print(f"  CLASSIFICATION : {best.upper()}")
        print(f"  votes          : {counts[best]}/{len(self.votes)}  "
              f"avg_conf={avg_conf:.2f}")
        print(f"  TARGET TAG     : {LANDING_TAG[best]}")
        print(f"{'='*40}\n")

        if frac < MIN_VOTE_FRACTION:
            print("[Stage4] Low agreement — collecting more frames.")
            self.votes   = []
            self.conf_sum = 0.0
            return

        self.class_label   = best
        self.target_tag_id = LANDING_TAG[best]
        self.votes         = []
        self.conf_sum      = 0.0
        self.stage         = Stage.NAVIGATING

    # ─── Control step ─────────────────────────────────────────────────────────

    def _control_step(self):
        if self.stage == Stage.IDLE:
            self._stop()
        elif self.stage == Stage.CLASSIFYING:
            self._stop()    # hover while classifying
        elif self.stage == Stage.NAVIGATING:
            self._navigate()
        elif self.stage == Stage.APPROACHING:
            self._approach()
        elif self.stage == Stage.LANDING:
            self._stop()
            self.tello.land()
            self.stage = Stage.IDLE

    def _navigate(self):
        if self.tag_in_view is not None:
            print(f"[Stage4] Tag {self.target_tag_id} found — switching to APPROACHING")
            for pid in (self.pid_fwd, self.pid_lat, self.pid_alt, self.pid_yaw):
                pid.reset()
            self.stage = Stage.APPROACHING
            return
        # Spin slowly to search
        self.tello.send_rc_control(0, 0, 0, 30)

    def _approach(self):
        if self.tag_in_view is None:
            print("[Stage4] Tag lost — back to NAVIGATING")
            self._stop()
            self.stage = Stage.NAVIGATING
            return

        t  = self.tag_in_view
        tx = float(t.pose_t[0])   # lateral offset (positive = tag right)
        ty = float(t.pose_t[1])   # vertical offset (positive = tag below)
        tz = float(t.pose_t[2])   # forward distance to tag

        err_fwd = tz - APPROACH_DISTANCE
        err_lat = tx
        err_alt = ty

        # Yaw alignment from tag rotation matrix
        rot     = t.pose_R
        yaw_err = math.atan2(rot[1, 0], rot[0, 0])

        # PID → RC percentages (-100 to 100)
        def to_rc(v): return int(np.clip(v * 100, -100, 100))

        fb  = to_rc( self.pid_fwd.compute(err_fwd))
        lr  = to_rc(-self.pid_lat.compute(err_lat))   # negate: right → move right
        ud  = to_rc(-self.pid_alt.compute(err_alt))   # negate: below → move up
        yaw = to_rc(-self.pid_yaw.compute(yaw_err))

        self.tello.send_rc_control(lr, fb, ud, yaw)

        # Landing condition
        if (abs(err_fwd) < LAND_POS_TOL and
                abs(err_lat) < LAND_LAT_TOL and
                abs(err_alt) < LAND_LAT_TOL):
            print("[Stage4] In position — landing")
            self._stop()
            self.stage = Stage.LANDING

    def _stop(self):
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

    # ─── Pygame ───────────────────────────────────────────────────────────────

    def _handle_pygame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c and self.stage == Stage.IDLE:
                    print("[Stage4] Starting classification — hold image in front of camera")
                    self.votes = []; self.conf_sum = 0.0
                    self.class_label = None; self.target_tag_id = None
                    self.stage = Stage.CLASSIFYING
                elif event.key == pygame.K_SPACE:
                    self._stop(); self.stage = Stage.IDLE
                elif event.key == pygame.K_l:
                    self._stop(); self.tello.land(); self.stage = Stage.IDLE
                elif event.key == pygame.K_r:
                    self._stop(); self.stage = Stage.IDLE

    def _render(self):
        BG    = (28,  30,  42)
        WHITE = (220, 222, 235)
        CYAN  = ( 80, 210, 255)
        GREEN = ( 90, 220, 120)
        YLW   = (255, 210,  60)
        GRAY  = (120, 122, 140)
        RED   = (255,  80,  80)

        stage_color = {
            Stage.IDLE: GRAY, Stage.CLASSIFYING: YLW,
            Stage.NAVIGATING: CYAN, Stage.APPROACHING: GREEN,
            Stage.LANDING: RED,
        }[self.stage]

        self.screen.fill(BG)
        y = 14

        def line(text, color=WHITE, font=None):
            nonlocal y
            surf = (font or self.font_m).render(text, True, color)
            self.screen.blit(surf, (14, y))
            y += surf.get_height() + 4

        line("  Stage 4 — Image Classifier & Lander", WHITE, self.font_l)
        line("-" * 44, GRAY, self.font_s)
        line(f"  State      : {self.stage.name}", stage_color)
        line(f"  Class      : {self.class_label or '---'}",
             GREEN if self.class_label else GRAY)
        line(f"  Target Tag : {self.target_tag_id or '---'}", WHITE)
        line(f"  Tag in view: {'YES' if self.tag_in_view else 'no'}",
             GREEN if self.tag_in_view else RED)
        line(f"  Votes      : {len(self.votes)}/{CLASSIFY_FRAMES}",
             YLW if self.stage == Stage.CLASSIFYING else GRAY)
        if self.tag_in_view is not None:
            t  = self.tag_in_view
            tz = float(t.pose_t[2])
            line(f"  Distance   : {tz:.2f} m  (target {APPROACH_DISTANCE} m)", CYAN)
        line("-" * 44, GRAY, self.font_s)
        line("  [C] Classify  [SPACE] Stop  [L] Land  [R] Reset",
             GRAY, self.font_s)
        pygame.display.flip()
