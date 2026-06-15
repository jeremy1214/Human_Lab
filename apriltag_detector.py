"""
apriltag_detector.py
====================
AprilTag detection from a live camera frame.
No ROS required — pure Python / OpenCV / pupil-apriltags.

Usage
-----
    detector = AprilTagDetector('Apriltag/map/apriltag_map.yaml')
    pose_6d  = detector.detect(bgr_frame)   # returns np.ndarray or None
"""

import os
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector


# ── Camera intrinsics (from Lab 1 calibration) ────────────────────────────────
FX, FY = 835.342103847164, 839.4691450667409
CX, CY = 415.5366635247159, 355.11975613817964
TAG_SIZE = 0.165   # metres


class AprilTagDetector:
    """
    Detects AprilTags in a BGR frame and returns the camera's world pose.

    Parameters
    ----------
    map_yaml : str
        Path to apriltag_map.yaml containing tag world positions.
    tag_ids : list[int] | None
        If given, only consider these tag IDs.  None = use all tags in the map.
    """

    def __init__(self, map_yaml: str, tag_ids: list = None):
        self.fx, self.fy = FX, FY
        self.cx, self.cy = CX, CY
        self.tag_size    = TAG_SIZE
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]

        self._detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.tag_pose_dict: dict = {}
        self._load_map(map_yaml, tag_ids)

        # Camera matrix for solvePnP
        self._camera_matrix = np.array(
            [[self.fx, 0, self.cx],
             [0, self.fy, self.cy],
             [0,       0,      1]], dtype=np.float32
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Local corners of a tag (centre = origin, tag plane = XY, normal = +Z)
        s = self.tag_size / 2.0
        self._local_corners = np.array(
            [[-s,  s, 0, 1],
             [ s,  s, 0, 1],
             [ s, -s, 0, 1],
             [-s, -s, 0, 1]], dtype=np.float32
        ).T

    # ─── Public API ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray):
        """
        Run detection on a BGR frame.

        Returns
        -------
        np.ndarray shape (6,)  [x, y, z, roll, yaw, pitch]  in world frame
        or None if no reliable detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )
        # Only keep tags that are in the map
        valid = [t for t in tags if t.tag_id in self.tag_pose_dict]
        if not valid:
            return None

        if len(valid) <= 2:
            return self._single_tag(valid)
        else:
            return self._multi_tag(valid)

    def detect_specific(self, frame: np.ndarray, tag_id: int):
        """
        Detect one specific tag and return its camera-frame pose (T_c_t, tag).
        Used by the Stage-4 approach controller.

        Returns (T_c_t 4×4, tag_detection) or (None, None).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )
        for t in tags:
            if t.tag_id == tag_id:
                T_c_t = np.eye(4)
                T_c_t[:3, :3] = t.pose_R
                T_c_t[:3,  3] = t.pose_t.flatten()
                return T_c_t, t
        return None, None

    def get_tag_world_pose(self, tag_id: int):
        """Return 4×4 world pose of a map tag, or None."""
        tag = self.tag_pose_dict.get(tag_id)
        if tag is None:
            return None
        pos = tag['position']
        rpy = tag['orientation_rpy']
        T   = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
        T[:3,  3] = pos
        return T

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _load_map(self, path: str, tag_ids):
        if not os.path.exists(path):
            print(f"[AprilTagDetector] WARNING: map file not found: {path}")
            return
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for tag in data.get('tags', []):
            tid = tag['id']
            if tag_ids is None or tid in tag_ids:
                self.tag_pose_dict[tid] = tag
        print(f"[AprilTagDetector] Loaded {len(self.tag_pose_dict)} tags from {path}")

    @staticmethod
    def _opencv_to_ros(T_w_c_cv: np.ndarray) -> np.ndarray:
        """Convert OpenCV world-camera transform to ROS/ENU frame."""
        R_cv_to_ros = np.array([
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ], dtype=float)
        return T_w_c_cv @ R_cv_to_ros

    def _T_to_pose6d(self, T: np.ndarray) -> np.ndarray:
        """Convert 4×4 homogeneous matrix → [x, y, z, roll, yaw, pitch]."""
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        rpy = rot.as_euler('xyz')   # roll, pitch, yaw
        return np.array([pos[0], pos[1], pos[2], rpy[0], rpy[2], rpy[1]])

    def _single_tag(self, tags: list):
        """Use the closest tag for a single-tag pose estimate."""
        best = min(tags, key=lambda t: float(np.linalg.norm(t.pose_t)))
        T_w_t = self.get_tag_world_pose(best.tag_id)
        if T_w_t is None:
            return None

        T_c_t = np.eye(4)
        T_c_t[:3, :3] = best.pose_R
        T_c_t[:3,  3] = best.pose_t.flatten()

        T_w_c = self._opencv_to_ros(T_w_t @ np.linalg.inv(T_c_t))
        return self._T_to_pose6d(T_w_c)

    def _multi_tag(self, tags: list):
        """Use solvePnPRansac + refine for multi-tag accuracy."""
        obj_pts, img_pts = [], []
        s = self.tag_size / 2.0

        for tag in tags:
            T_w_t = self.get_tag_world_pose(tag.tag_id)
            if T_w_t is None:
                continue
            world_corners = (T_w_t @ self._local_corners)[:3, :].T
            obj_pts.extend(world_corners)
            img_pts.extend(tag.corners)

        if len(obj_pts) < 4:
            return self._single_tag(tags)

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0, confidence=0.99, iterationsCount=100,
        )
        if not ok or inliers is None or len(inliers) < 4:
            return self._single_tag(tags)

        idx = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts[idx], img_pts[idx],
            self._camera_matrix, self._dist_coeffs, rvec, tvec,
        )

        # Reprojection error check
        proj, _ = cv2.projectPoints(obj_pts[idx], rvec, tvec,
                                     self._camera_matrix, self._dist_coeffs)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts[idx], axis=1)))
        if err > 5.0:
            return None

        R_c_w, _ = cv2.Rodrigues(rvec)
        T_c_w = np.eye(4)
        T_c_w[:3, :3] = R_c_w
        T_c_w[:3,  3] = tvec.flatten()

        T_w_c = self._opencv_to_ros(np.linalg.inv(T_c_w))
        return self._T_to_pose6d(T_w_c)
