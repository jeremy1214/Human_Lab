"""
ekf_localization.py
===================
Extended Kalman Filter for Tello drone pose estimation.
No ROS required — pure Python / NumPy.

State vector  : [x, y, z, roll, yaw, pitch]   (metres, radians)
Control vector: [vx, vy, vz, roll_rate, yaw_rate, pitch_rate]
Measurement   : same layout as state (from AprilTag detections)
"""

import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R


class EKFLocalization:
    """Thread-safe EKF.  Call step() from a timer thread; feed detections via update_from_detection()."""

    def __init__(self):
        # ── State & covariance ────────────────────────────────────────────────
        self.mu    = np.zeros((6, 1))      # [x, y, z, roll, yaw, pitch]
        self.Sigma = np.eye(6) * 0.1

        # ── Noise matrices ────────────────────────────────────────────────────
        self.Rm = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])   # process noise
        self.Q  = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])   # measurement noise

        # ── Control input (updated externally) ───────────────────────────────
        self._u    = np.zeros((6, 1))
        self._lock = threading.Lock()

        # ── Initialisation guard ──────────────────────────────────────────────
        self._init_count = 0
        self.is_initialized = False
        self._reject_count  = 0

    # ─── Public getters ───────────────────────────────────────────────────────

    @property
    def pose(self) -> np.ndarray:
        """Return current best-estimate pose as a 1-D numpy array [x,y,z,roll,yaw,pitch]."""
        with self._lock:
            return self.mu.flatten().copy()

    @property
    def position(self):
        p = self.pose
        return p[0], p[1], p[2]

    @property
    def yaw_rad(self) -> float:
        return float(self.pose[4])

    # ─── Control input ────────────────────────────────────────────────────────

    def set_control(self, vx: float, vy: float, vz: float,
                    roll_rate: float, yaw_rate: float, pitch_rate: float):
        """Update the control vector (call this from the state-polling thread)."""
        with self._lock:
            self._u = np.array([[vx], [vy], [vz],
                                 [roll_rate], [yaw_rate], [pitch_rate]])

    # ─── EKF step (call at ~10 Hz) ────────────────────────────────────────────

    def step(self, dt: float):
        """Run one predict cycle using the current control input."""
        with self._lock:
            if dt <= 0.0 or dt >= 1.0:
                return
            self._predict(self._u, dt)

    # ─── Measurement update (called from vision thread) ──────────────────────

    def update_from_detection(self, pose_6d: np.ndarray):
        """
        Feed an AprilTag-derived pose measurement into the EKF.

        pose_6d : np.ndarray shape (6,) or (6,1)  [x, y, z, roll, yaw, pitch]
        """
        z = np.asarray(pose_6d, dtype=float).reshape(6, 1)
        with self._lock:
            if not self.is_initialized:
                self._init_count += 1
                self._update(z)
                if self._init_count > 5:
                    self.is_initialized = True
                return

            # Outlier rejection
            pred = self.mu[:3, 0]
            obs  = z[:3, 0]
            dist = float(np.linalg.norm(obs - pred))
            if dist > 2.0:
                self._reject_count += 1
                if self._reject_count > 5:
                    # EKF lost — hard reset to observation
                    self.mu[:3, 0]  = z[:3, 0]
                    self.Sigma      = np.eye(6) * 0.5
                    self._reject_count = 0
                    self._update(z)
                return

            self._reject_count = 0
            self._update(z)

    # ─── Internal EKF maths (called while lock is held) ──────────────────────

    def _motion_model(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        xp = x.copy()
        yaw = float(x[4, 0])
        vx, vy, vz          = u[0,0], u[1,0], u[2,0]
        roll_r, yaw_r, pit_r = u[3,0], u[4,0], u[5,0]

        cy, sy = math.cos(yaw), math.sin(yaw)
        xp[0, 0] += (vx * cy - vy * sy) * dt
        xp[1, 0] += (vx * sy + vy * cy) * dt
        xp[2, 0] += vz * dt
        xp[3, 0] += roll_r * dt
        xp[4, 0] += yaw_r  * dt
        xp[5, 0] += pit_r  * dt

        for i in range(3, 6):
            xp[i, 0] = (xp[i, 0] + math.pi) % (2 * math.pi) - math.pi
        return xp

    def _jacobian_F(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        F   = np.eye(6)
        yaw = float(x[4, 0])
        vx, vy = u[0, 0], u[1, 0]
        cy, sy = math.cos(yaw), math.sin(yaw)
        F[0, 4] = (-vx * sy - vy * cy) * dt
        F[1, 4] = ( vx * cy - vy * sy) * dt
        return F

    def _predict(self, u: np.ndarray, dt: float):
        self.mu    = self._motion_model(self.mu, u, dt)
        F          = self._jacobian_F(self.mu, u, dt)
        self.Sigma = F @ self.Sigma @ F.T + self.Rm

    def _update(self, z: np.ndarray):
        C = np.eye(6)
        y = z - C @ self.mu
        for i in range(3, 6):
            y[i, 0] = (y[i, 0] + math.pi) % (2 * math.pi) - math.pi

        S = C @ self.Sigma @ C.T + self.Q      # ← bug-fix: was inv(5)
        K = self.Sigma @ C.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        for i in range(3, 6):
            self.mu[i, 0] = (self.mu[i, 0] + math.pi) % (2 * math.pi) - math.pi
        self.Sigma = (np.eye(6) - K @ C) @ self.Sigma
