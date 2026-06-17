#!/usr/bin/env python3
"""
image_classifier_node.py  —  Stage 4 Landing System
=====================================================
Classifies a brainrot meme held in front of Tello's camera (YOLOv8 ONNX),
then navigates to and lands in front of the matching AprilTag landing zone.

Class → Landing Zone mapping (from competition spec):
  cap   → Zone 1  (AprilTag id 13)
  brr   → Zone 2  (AprilTag id 14)
  trala → Zone 3  (AprilTag id 15)
  tung  → Zone 4  (AprilTag id 16)

State machine:
  IDLE ──[C key]──► CLASSIFYING ──[vote done]──► NAVIGATING ──[tag visible]──► APPROACHING ──[close enough]──► LANDING

Keybindings (pygame window must be focused):
  [C]     : begin image classification
  [SPACE] : emergency stop & back to IDLE
  [L]     : force-land immediately
  [R]     : reset / go back to IDLE without landing
"""

import enum
import math
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import pygame
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from pupil_apriltags import Detector
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Class indices match the YOLO yaml: {0: cap, 1: brr, 2: trala, 3: tung}
CLASS_NAMES = ['cap', 'brr', 'trala', 'tung']

# Which landing AprilTag corresponds to each class
LANDING_TAG = {
    'cap':   13,   # Landing Zone 1
    'brr':   14,   # Landing Zone 2
    'trala': 15,   # Landing Zone 3
    'tung':  16,   # Landing Zone 4
}

# YOLO inference settings
YOLO_INPUT_SIZE   = 640
YOLO_CONF_THRESH  = 0.15   # same threshold used during training validation
YOLO_NMS_THRESH   = 0.45

# How many frames to accumulate before declaring a classification result
CLASSIFY_FRAMES   = 15
MIN_VOTE_FRACTION = 0.5    # at least 50 % of frames must agree

# Approach geometry
APPROACH_DISTANCE = 0.55   # m – want tag this far away (mid of 40–70 cm range)
LAND_DIST_THRESH  = 0.10   # m – positional tolerance to trigger land
LAND_ALIGN_THRESH = 0.08   # m – lateral/vertical tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Stage enum
# ─────────────────────────────────────────────────────────────────────────────

class Stage(enum.IntEnum):
    IDLE        = 0
    CLASSIFYING = 1
    NAVIGATING  = 2
    APPROACHING = 3
    LANDING     = 4


# ─────────────────────────────────────────────────────────────────────────────
# Simple PID controller
# ─────────────────────────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp: float, ki: float, kd: float, limit: float = 1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self._integral  = 0.0
        self._prev_err  = 0.0

    def compute(self, error: float, dt: float = 0.05) -> float:
        self._integral += error * dt
        derivative      = (error - self._prev_err) / max(dt, 1e-6)
        self._prev_err  = error
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(out, -self.limit, self.limit))

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# YOLO post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, new_size: int = 640):
    """Resize with aspect-ratio-preserving letterboxing."""
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad_y, pad_x = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized
    return canvas, scale, pad_x, pad_y


def preprocess_yolo(frame: np.ndarray):
    """Prepare a BGR frame for YOLOv8 ONNX inference."""
    lb, scale, pad_x, pad_y = _letterbox(frame, YOLO_INPUT_SIZE)
    rgb  = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))          # HWC → CHW
    blob = np.expand_dims(blob, axis=0)            # → [1, 3, 640, 640]
    return blob, scale, pad_x, pad_y


def postprocess_yolo(output: np.ndarray,
                     conf_thresh: float = YOLO_CONF_THRESH,
                     nms_thresh:  float = YOLO_NMS_THRESH):
    """
    Parse YOLOv8 ONNX output [1, 8, 8400] into detections.

    YOLOv8 output layout (axis-1):
      [0:4]  → x_c, y_c, w, h  (pixel coords at 640×640 scale)
      [4:8]  → raw class scores for cap, brr, trala, tung

    Returns list of dicts: {'class_id', 'class_name', 'confidence', 'box'}
    """
    raw  = output[0]                    # [8, 8400]
    rows = raw.T                        # [8400, 8]

    boxes_xywh = rows[:, :4]           # centre-x, centre-y, w, h
    class_scores = rows[:, 4:]         # [8400, 4]

    class_ids    = np.argmax(class_scores, axis=1)
    confidences  = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences >= conf_thresh
    if not np.any(mask):
        return []

    boxes_f  = boxes_xywh[mask]
    confs_f  = confidences[mask]
    class_f  = class_ids[mask]

    # Convert xywh → x1y1x2y2 for NMS
    x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
    y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
    x2 = boxes_f[:, 0] + boxes_f[:, 2] / 2
    y2 = boxes_f[:, 1] + boxes_f[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # OpenCV NMS (per class)
    detections = []
    for cid in np.unique(class_f):
        idx_c  = np.where(class_f == cid)[0]
        bxs_c  = boxes_xyxy[idx_c].tolist()
        cfs_c  = confs_f[idx_c].tolist()
        keep   = cv2.dnn.NMSBoxes(
            [[x, y, x2 - x, y2 - y] for x, y, x2, y2 in bxs_c],
            cfs_c, conf_thresh, nms_thresh
        )
        if len(keep) == 0:
            continue
        for k in (keep.flatten() if isinstance(keep, np.ndarray) else keep):
            detections.append({
                'class_id':   int(cid),
                'class_name': CLASS_NAMES[int(cid)],
                'confidence': float(cfs_c[k]),
                'box':        bxs_c[k],   # [x1, y1, x2, y2] at 640-px scale
            })

    return sorted(detections, key=lambda d: d['confidence'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────────────────────────

class ImageClassifierNode(Node):
    """
    Stage 4 node: classify brainrot image → navigate → approach AprilTag → land.
    """

    def __init__(self):
        super().__init__('image_classifier_node')

        # ── Camera / CV bridge ───────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── AprilTag detector (for landing-zone tags 13-16) ─────────────────
        self.at_detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        # Camera intrinsics (same as apriltag_detector_node.py)
        self.fx, self.fy = 835.342103847164,  839.4691450667409
        self.cx, self.cy = 415.5366635247159, 355.11975613817964
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]
        # NOTE: If landing-zone AprilTags have a different physical size than
        #       the localisation tags (0.165 m), update this value.
        self.tag_size = 0.165

        # ── Load YOLO ONNX model ─────────────────────────────────────────────
        model_path = self._find_model('brainrot_detect.onnx')
        self.get_logger().info(f"Loading ONNX model from: {model_path}")
        self.ort_session  = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
        self.ort_input_name = self.ort_session.get_inputs()[0].name
        self.get_logger().info("ONNX model loaded successfully.")

        # ── ROS Subscribers / Publishers ─────────────────────────────────────
        self.img_sub  = self.create_subscription(
            Image, '/image_raw', self._image_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/ekf_pose', self._pose_callback, 10
        )
        self.vel_pub  = self.create_publisher(Twist, 'cmd_vel', 10)

        # ── Tello services (via tello_interface_node) ────────────────────────
        self.land_client      = self.create_client(Trigger, '/tello/land')
        self.emergency_client = self.create_client(Trigger, '/tello/emergency')

        # ── State machine ────────────────────────────────────────────────────
        self.stage          = Stage.IDLE
        self.class_label    = None       # e.g. 'cap'
        self.target_tag_id  = None       # e.g. 13
        self.tag_in_view    = None       # pupil_apriltags detection object
        self.current_pose   = None       # PoseWithCovarianceStamped.pose.pose

        # Classification vote buffer
        self.classify_votes: list[str] = []
        self.classify_conf_sum: float  = 0.0

        # ── PID controllers (tuned conservatively) ───────────────────────────
        # forward (+x in camera frame = closer to tag)
        self.pid_fwd = PID(kp=0.4, ki=0.0, kd=0.08, limit=0.25)
        # lateral  (+y in camera frame = tag to the right → fly right)
        self.pid_lat = PID(kp=0.5, ki=0.0, kd=0.10, limit=0.20)
        # vertical (+y in camera frame = tag below → fly up)
        self.pid_alt = PID(kp=0.4, ki=0.0, kd=0.08, limit=0.20)
        # yaw alignment
        self.pid_yaw = PID(kp=0.5, ki=0.0, kd=0.05, limit=0.35)

        # ── Pygame status window ─────────────────────────────────────────────
        pygame.init()
        self.screen = pygame.display.set_mode((520, 400))
        pygame.display.set_caption('Stage 4 — Image Classifier & Landing')
        self.font_l = pygame.font.SysFont('monospace', 22, bold=True)
        self.font_m = pygame.font.SysFont('monospace', 19)
        self.font_s = pygame.font.SysFont('monospace', 15)

        # ── Main 20 Hz control timer ─────────────────────────────────────────
        self.ctrl_timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info(
            "ImageClassifierNode ready.  "
            "Focus the pygame window and press [C] to start."
        )

    # ─── Model path resolver ─────────────────────────────────────────────────

    def _find_model(self, filename: str) -> str:
        """Search for the ONNX model in common locations."""
        candidates = [
            # Installed into ROS share directory
            os.path.join(
                get_package_share_directory('tello_localization'),
                'model', filename
            ),
            # Beside this source file
            os.path.join(os.path.dirname(__file__), filename),
            # Current working directory
            os.path.join(os.getcwd(), filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Cannot find '{filename}'. "
            f"Tried: {candidates}. "
            "Copy the model into the package's 'model/' directory."
        )

    # ─── Tello helper ────────────────────────────────────────────────────────

    def _send_tello_cmd(self, cmd: str):
        """Send 'land' or 'emergency' via the corresponding Trigger service."""
        client = {'land': self.land_client, 'emergency': self.emergency_client}.get(cmd)
        if client is None:
            self.get_logger().warn(f"Unknown tello cmd: {cmd}")
            return
        if not client.service_is_ready():
            self.get_logger().warn(f"/tello/{cmd} service not ready – skipping")
            return
        client.call_async(Trigger.Request())
        self.get_logger().info(f"Tello cmd → {cmd}")

    def _stop(self):
        self.vel_pub.publish(Twist())

    # ─── ROS Callbacks ───────────────────────────────────────────────────────

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg.pose.pose

    def _image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ── Always: try to detect the target landing-zone AprilTag ──────────
        self.tag_in_view = None
        if self.target_tag_id is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size,
            )
            for t in tags:
                if t.tag_id == self.target_tag_id:
                    self.tag_in_view = t
                    break

        # ── CLASSIFYING: run YOLO inference and accumulate votes ─────────────
        if self.stage == Stage.CLASSIFYING:
            label, conf = self._classify(frame)
            if label is not None:
                self.classify_votes.append(label)
                self.classify_conf_sum += conf

            if len(self.classify_votes) >= CLASSIFY_FRAMES:
                self._finalise_classification()

    # ─── YOLO Inference ──────────────────────────────────────────────────────

    def _classify(self, frame: np.ndarray):
        """
        Run YOLOv8 on one frame and return (best_class_name, confidence)
        or (None, 0) if no detection passes the threshold.
        """
        blob, _, _, _ = preprocess_yolo(frame)
        output = self.ort_session.run(None, {self.ort_input_name: blob})
        detections = postprocess_yolo(output[0])

        if not detections:
            return None, 0.0

        best = detections[0]    # already sorted by confidence desc
        return best['class_name'], best['confidence']

    def _finalise_classification(self):
        """Tally votes, log result, and move to NAVIGATING."""
        if not self.classify_votes:
            self.get_logger().warn("No valid classification frames — staying in CLASSIFYING.")
            self.classify_votes = []
            return

        vote_counts = {cls: self.classify_votes.count(cls) for cls in CLASS_NAMES}
        best_label  = max(vote_counts, key=vote_counts.get)
        vote_frac   = vote_counts[best_label] / len(self.classify_votes)
        avg_conf    = self.classify_conf_sum / len(self.classify_votes)

        self.get_logger().info(
            f"Classification result: {best_label.upper()}  "
            f"(votes {vote_counts[best_label]}/{len(self.classify_votes)}, "
            f"avg_conf={avg_conf:.2f})"
        )

        if vote_frac < MIN_VOTE_FRACTION:
            self.get_logger().warn(
                f"Low vote agreement ({vote_frac:.0%}) — "
                "collecting more frames."
            )
            self.classify_votes = []
            self.classify_conf_sum = 0.0
            return

        self.class_label   = best_label
        self.target_tag_id = LANDING_TAG[best_label]
        self.classify_votes = []
        self.classify_conf_sum = 0.0
        self.stage = Stage.NAVIGATING

        # Print result to terminal (satisfies Cp3-R1)
        print(f"\n{'='*40}")
        print(f"  CLASSIFICATION: {self.class_label}")
        print(f"  TARGET AprilTag: {self.target_tag_id}")
        print(f"{'='*40}\n")

    # ─── Main Control Loop ────────────────────────────────────────────────────

    def _control_loop(self):
        self._handle_pygame()
        self._render_pygame()

        if self.stage == Stage.IDLE:
            self._stop()

        elif self.stage == Stage.CLASSIFYING:
            self._stop()           # hover while classifying

        elif self.stage == Stage.NAVIGATING:
            self._navigate()

        elif self.stage == Stage.APPROACHING:
            self._approach()

        elif self.stage == Stage.LANDING:
            self._stop()
            self._send_tello_cmd('land')
            self.stage = Stage.IDLE

    # ─── Navigation: spin to find landing-zone tag ───────────────────────────

    def _navigate(self):
        """
        Coarse search: rotate slowly until the target AprilTag becomes visible,
        then switch to APPROACHING.
        """
        if self.tag_in_view is not None:
            self.get_logger().info(
                f"Landing tag {self.target_tag_id} acquired — switching to APPROACHING."
            )
            # Reset PIDs for a fresh approach
            for pid in (self.pid_fwd, self.pid_lat, self.pid_alt, self.pid_yaw):
                pid.reset()
            self.stage = Stage.APPROACHING
            return

        # Rotate slowly counter-clockwise to scan
        twist = Twist()
        twist.angular.z = 0.30      # rad/s  — slow rotation
        self.vel_pub.publish(twist)

    # ─── Approach: PID to fly to APPROACH_DISTANCE in front of tag ───────────

    def _approach(self):
        """
        Fine approach using camera-frame AprilTag pose.

        OpenCV camera frame (what pupil_apriltags gives):
          +X → right,  +Y → down,  +Z → into the scene (forward)

        Tello /cmd_vel convention:
          linear.x  → forward(+) / backward(−)
          linear.y  → left(+) / right(−)
          linear.z  → up(+) / down(−)
          angular.z → CCW yaw(+) / CW yaw(−)
        """
        if self.tag_in_view is None:
            # Tag lost — fall back to scanning
            self.get_logger().warn("Lost tag during approach — back to NAVIGATING.")
            self._stop()
            self.stage = Stage.NAVIGATING
            return

        tag = self.tag_in_view

        # Camera-frame translation to tag centre
        tx = float(tag.pose_t[0])   # positive → tag is to the right
        ty = float(tag.pose_t[1])   # positive → tag is below centre
        tz = float(tag.pose_t[2])   # positive → tag is in front

        # Error definitions (positive error → positive control output → drone moves toward tag)
        err_fwd  = tz - APPROACH_DISTANCE   # >0: still too far
        err_lat  = tx                        # >0: tag to the right → move right (−y in ROS)
        err_alt  = ty                        # >0: tag below → move up (+z in ROS)

        # Yaw alignment: use the Z-axis of the tag's rotation matrix
        # We want the tag to appear "straight on" (yaw error ≈ 0)
        rot = tag.pose_R
        yaw_err = math.atan2(rot[1, 0], rot[0, 0])   # approx yaw offset

        # Compute PID outputs
        vx   =  self.pid_fwd.compute(err_fwd)
        vy   = -self.pid_lat.compute(err_lat)   # negate: right → move right (−y)
        vz   = -self.pid_alt.compute(err_alt)   # negate: below → move up
        vyaw = -self.pid_yaw.compute(yaw_err)

        twist = Twist()
        twist.linear.x  = float(np.clip(vx,   -0.25, 0.25))
        twist.linear.y  = float(np.clip(vy,   -0.20, 0.20))
        twist.linear.z  = float(np.clip(vz,   -0.20, 0.20))
        twist.angular.z = float(np.clip(vyaw, -0.35, 0.35))
        self.vel_pub.publish(twist)

        # Debug log
        self.get_logger().info(
            f"tag {self.target_tag_id} | "
            f"fwd_err={err_fwd:.3f} lat_err={err_lat:.3f} "
            f"alt_err={err_alt:.3f} yaw_err={math.degrees(yaw_err):.1f}°",
            throttle_duration_sec=0.5,
        )

        # Landing condition: centred and at the right distance
        if (abs(err_fwd)  < LAND_DIST_THRESH and
                abs(err_lat) < LAND_ALIGN_THRESH and
                abs(err_alt) < LAND_ALIGN_THRESH):
            self.get_logger().info("In landing position — switching to LANDING.")
            self._stop()
            self.stage = Stage.LANDING

    # ─── Pygame ──────────────────────────────────────────────────────────────

    def _handle_pygame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_c:
                    if self.stage == Stage.IDLE:
                        self.get_logger().info("Starting classification — hold image in front of camera.")
                        self.class_label       = None
                        self.target_tag_id     = None
                        self.classify_votes    = []
                        self.classify_conf_sum = 0.0
                        self.stage = Stage.CLASSIFYING
                    else:
                        self.get_logger().warn("[C] ignored — not in IDLE state.")

                elif event.key == pygame.K_SPACE:
                    self.get_logger().warn("EMERGENCY STOP — returning to IDLE.")
                    self._stop()
                    self.stage = Stage.IDLE

                elif event.key == pygame.K_l:
                    self.get_logger().info("Force land.")
                    self._stop()
                    self._send_tello_cmd('land')
                    self.stage = Stage.IDLE

                elif event.key == pygame.K_r:
                    self.get_logger().info("Reset to IDLE.")
                    self._stop()
                    self.stage = Stage.IDLE

    def _render_pygame(self):
        BG    = (28,  30,  42)
        WHITE = (220, 222, 235)
        CYAN  = ( 80, 210, 255)
        GREEN = ( 90, 220, 120)
        YLW   = (255, 210,  60)
        RED   = (255,  80,  80)
        GRAY  = (120, 122, 140)

        stage_color = {
            Stage.IDLE:        GRAY,
            Stage.CLASSIFYING: YLW,
            Stage.NAVIGATING:  CYAN,
            Stage.APPROACHING: GREEN,
            Stage.LANDING:     RED,
        }[self.stage]

        tag_status   = f"YES  (id {self.tag_in_view.tag_id})" if self.tag_in_view else "not visible"
        tag_color    = GREEN if self.tag_in_view else RED
        label_str    = self.class_label.upper() if self.class_label else '---'
        target_str   = str(self.target_tag_id) if self.target_tag_id else '---'
        votes_str    = f"{len(self.classify_votes)}/{CLASSIFY_FRAMES}" if self.stage == Stage.CLASSIFYING else '—'

        self.screen.fill(BG)
        y = 16
        def line(text, color, font=None):
            nonlocal y
            surf = (font or self.font_m).render(text, True, color)
            self.screen.blit(surf, (16, y))
            y += surf.get_height() + 4

        line("  Stage 4 — Image Classifier & Lander", WHITE, self.font_l)
        line("─" * 44, GRAY, self.font_s)
        line(f"  State      : {self.stage.name:<14}", stage_color)
        line(f"  Class      : {label_str}", GREEN if self.class_label else GRAY)
        line(f"  Target Tag : {target_str}", WHITE)
        line(f"  Tag in view: {tag_status}", tag_color)
        line(f"  Votes      : {votes_str}", YLW if self.stage == Stage.CLASSIFYING else GRAY)
        line("─" * 44, GRAY, self.font_s)
        line("  [C] Classify   [SPACE] Stop   [L] Land   [R] Reset", GRAY, self.font_s)

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ImageClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()