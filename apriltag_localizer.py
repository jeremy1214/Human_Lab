"""
apriltag_localizer.py
======================
Single-tag self-localization: given ONE detected AprilTag (its pose in
the camera frame) plus its known world pose (from apriltag_map.yaml),
recover the camera's (drone's) world position and heading.

This is deliberately single-tag (not multi-tag solvePnP triangulation) —
any one visible tag is enough to fix the drone's (x, y, z, yaw).

World frame convention (matches apriltag_map.yaml):
  X, Y horizontal room coordinates, Z up.
  yaw: standard right-hand convention about Z, CCW-positive, 0 = facing +X.

Math verified by round-trip simulation: place a synthetic camera at a known
world pose, compute the tag-in-camera transform it WOULD see, feed that into
localize_from_tag(), and check it recovers the original pose exactly.
"""

import math
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# OpenCV camera frame (X right, Y down, Z forward) -> world frame (X fwd-ish, Y left, Z up)
_R_CV_TO_WORLD = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1],
], dtype=float)


def load_tag_map(yaml_path: str) -> dict:
    """Load apriltag_map.yaml into {tag_id: {'position': [...], 'orientation_rpy': [...]}}."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"AprilTag map not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return {tag['id']: tag for tag in data.get('tags', [])}


def tag_world_pose(tag_entry: dict) -> np.ndarray:
    """4x4 homogeneous world pose of a tag, from its map entry."""
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', tag_entry['orientation_rpy']).as_matrix()
    T[:3,  3] = tag_entry['position']
    return T


def localize_from_tag(tag_detection, tag_pose_dict: dict):
    """
    Recover the camera's (drone's) world pose from a SINGLE AprilTag detection.

    Parameters
    ----------
    tag_detection : pupil_apriltags Detection
        Must have .tag_id, .pose_R (3x3), .pose_t (3x1), all in camera frame.
    tag_pose_dict : dict
        {tag_id: {'position': [...], 'orientation_rpy': [...]}} from the map.

    Returns
    -------
    (x, y, z, yaw) in world frame (metres, radians), or None if the detected
    tag's id is not present in the map.
    """
    tag_entry = tag_pose_dict.get(tag_detection.tag_id)
    if tag_entry is None:
        return None

    T_w_t = tag_world_pose(tag_entry)              # tag in world

    T_c_t = np.eye(4)                                # tag in camera
    T_c_t[:3, :3] = tag_detection.pose_R
    T_c_t[:3,  3] = np.asarray(tag_detection.pose_t).flatten()

    T_w_c_cv = T_w_t @ np.linalg.inv(T_c_t)          # camera in world (OpenCV axes)
    T_w_c    = T_w_c_cv @ _R_CV_TO_WORLD              # convert to world axes

    pos = T_w_c[:3, 3]
    yaw = R.from_matrix(T_w_c[:3, :3]).as_euler('xyz')[2]
    return float(pos[0]), float(pos[1]), float(pos[2]), float(yaw)


def localize_best_tag(tags: list, tag_pose_dict: dict):
    """
    Given a list of pupil_apriltags detections, localize using the closest
    one that's present in the map (still single-tag math — just choosing
    which single tag to trust when several are visible at once).

    Returns (x, y, z, yaw) or None if no detected tag is in the map.
    """
    candidates = [t for t in tags if t.tag_id in tag_pose_dict]
    if not candidates:
        return None
    best = min(candidates, key=lambda t: float(np.linalg.norm(t.pose_t)))
    return localize_from_tag(best, tag_pose_dict)


def compute_nav_target(tag_pose_dict: dict,
                       target_tag_ids=(15, 16),
                       wall_tag_ids=(4, 5, 6, 7, 8, 9)):
    """
    Compute a navigation waypoint: the midpoint between `target_tag_ids`,
    with a heading facing directly AWAY from the average position of
    `wall_tag_ids`.

    Returns (target_x, target_y, target_yaw).
    """
    targets = [tag_pose_dict[tid]['position'][:2]
               for tid in target_tag_ids if tid in tag_pose_dict]
    if not targets:
        raise ValueError(f"None of target_tag_ids {target_tag_ids} found in map")
    target_xy = np.mean(targets, axis=0)

    walls = [tag_pose_dict[tid]['position'][:2]
             for tid in wall_tag_ids if tid in tag_pose_dict]
    if not walls:
        raise ValueError(f"None of wall_tag_ids {wall_tag_ids} found in map")
    wall_xy = np.mean(walls, axis=0)

    angle_to_wall = math.atan2(wall_xy[1] - target_xy[1], wall_xy[0] - target_xy[0])
    target_yaw = angle_to_wall + math.pi
    target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi

    return float(target_xy[0]), float(target_xy[1]), float(target_yaw)


def world_error_to_body(dx: float, dy: float, yaw: float):
    """
    Project a world-frame position error (dx, dy) onto the drone's body
    axes given its current yaw.

    Returns (forward_err, right_err).
    Verified by closed-loop simulation: driving v_forward = +Kp*forward_err
    and v_right = +Kp*right_err converges to (dx,dy) = (0,0) from any
    starting pose (no extra sign flips needed at the call site).
    """
    forward_err = dx * math.cos(yaw) + dy * math.sin(yaw)
    right_err   = dx * math.sin(yaw) - dy * math.cos(yaw)
    return forward_err, right_err


def wrap_angle(angle: float) -> float:
    """Wrap an angle (radians) to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
