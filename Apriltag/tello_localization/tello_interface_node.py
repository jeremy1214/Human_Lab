#!/usr/bin/env python3
"""
tello_interface_node.py  —  djitellopy ↔ ROS 2 Bridge
=======================================================
Replaces the external `tello_driver` package.

Publishes
---------
/image_raw          sensor_msgs/Image          — live camera frames (BGR8)
/tello/flight_data  geometry_msgs/TwistStamped — velocity + attitude
                      .twist.linear   : vgx, vgy, vgz (cm/s, body FRD frame)
                      .twist.angular  : roll, pitch, yaw (degrees)

Subscribes
----------
/cmd_vel  geometry_msgs/Twist  — RC velocity commands
            .linear.x  : forward (+) / backward (−)  [−1.0 … 1.0  → −100 … 100 %]
            .linear.y  : left (+) / right (−)
            .linear.z  : up (+) / down (−)
            .angular.z : CCW yaw (+) / CW yaw (−)

Services
--------
/tello/takeoff   std_srvs/Trigger
/tello/land      std_srvs/Trigger
/tello/emergency std_srvs/Trigger

Parameters
----------
speed_scale (float, default 100.0)
    Divide Twist linear values by this to get RC % (1.0 → 100 %).
yaw_scale (float, default 100.0)
    Divide Twist angular.z by this to get RC % (1.0 → 100 %).
"""

import math
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from djitellopy import Tello
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

# Maximum RC value (Tello SDK: -100 … 100)
RC_MAX = 100


class TelloInterfaceNode(Node):
    """Bridge between djitellopy and ROS 2 topics/services."""

    def __init__(self):
        super().__init__('tello_interface_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('speed_scale', 100.0)   # divide Twist linear by this → RC %
        self.declare_parameter('yaw_scale',   100.0)   # divide Twist angular.z by this → RC %
        self.speed_scale = self.get_parameter('speed_scale').value
        self.yaw_scale   = self.get_parameter('yaw_scale').value

        # ── djitellopy Tello object ───────────────────────────────────────────
        self.tello = Tello()
        self.tello.connect()
        self.get_logger().info(
            f"Tello connected.  Battery: {self.tello.get_battery()} %"
        )
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()
        self.get_logger().info("Video stream started.")

        # ── Current RC command (updated by /cmd_vel, sent by timer) ───────────
        self._rc_lock = threading.Lock()
        self._rc      = [0, 0, 0, 0]   # [lr, fb, ud, yaw]

        # ── Bridge / publishers ───────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.img_pub         = self.create_publisher(Image,         '/image_raw',        1)
        self.flight_data_pub = self.create_publisher(TwistStamped,  '/tello/flight_data', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, 10)

        # ── Services ──────────────────────────────────────────────────────────
        self.create_service(Trigger, '/tello/takeoff',   self._takeoff_cb)
        self.create_service(Trigger, '/tello/land',      self._land_cb)
        self.create_service(Trigger, '/tello/emergency', self._emergency_cb)

        # ── Timers ────────────────────────────────────────────────────────────
        # 30 Hz camera publish
        self.create_timer(1.0 / 30.0, self._publish_frame)
        # 10 Hz RC control send
        self.create_timer(0.1,         self._send_rc)
        # 10 Hz state publish
        self.create_timer(0.1,         self._publish_flight_data)

        self.get_logger().info(
            "TelloInterfaceNode ready.\n"
            "  Services : /tello/takeoff  /tello/land  /tello/emergency\n"
            "  Topics   : /image_raw  /tello/flight_data  /cmd_vel"
        )

    # ─── Camera ───────────────────────────────────────────────────────────────

    def _publish_frame(self):
        frame = self.frame_read.frame         # BGR numpy array
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.img_pub.publish(msg)

    # ─── Flight data (velocity + attitude) ────────────────────────────────────

    def _publish_flight_data(self):
        """
        Read Tello state and publish as TwistStamped.

        Layout (velocity + attitude fields, matching the conversion expected by
        ekf_localization_node):
          .twist.linear  : vgx, vgy, vgz  [cm/s, body FRD frame]
          .twist.angular : roll, pitch, yaw [degrees]
        """
        try:
            msg = TwistStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'

            # Velocity (raw SDK field, historically treated as cm/s)
            msg.twist.linear.x = float(self.tello.get_state_field('vgx'))
            msg.twist.linear.y = float(self.tello.get_state_field('vgy'))
            msg.twist.linear.z = float(self.tello.get_state_field('vgz'))

            # Attitude (degrees)
            msg.twist.angular.x = float(self.tello.get_roll())
            msg.twist.angular.y = float(self.tello.get_pitch())
            msg.twist.angular.z = float(self.tello.get_yaw())

            self.flight_data_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Could not read Tello state: {e}", throttle_duration_sec=2.0)

    # ─── /cmd_vel → RC ───────────────────────────────────────────────────────

    def _cmd_vel_callback(self, msg: Twist):
        """
        Convert ROS Twist (FLU frame) to Tello RC values (−100 … 100).

        Twist convention (ROS body FLU):
          linear.x  : forward +
          linear.y  : left +
          linear.z  : up +
          angular.z : CCW yaw +

        djitellopy send_rc_control convention:
          left_right_velocity       : right +  → negate Twist.linear.y
          forward_backward_velocity : forward + → Twist.linear.x
          up_down_velocity          : up +      → Twist.linear.z
          yaw_velocity              : CW +      → negate Twist.angular.z
        """
        def to_rc(val: float, scale: float) -> int:
            return int(np.clip(val / scale * RC_MAX, -RC_MAX, RC_MAX))

        lr  = to_rc(-msg.linear.y,  self.speed_scale)
        fb  = to_rc( msg.linear.x,  self.speed_scale)
        ud  = to_rc( msg.linear.z,  self.speed_scale)
        yaw = to_rc(-msg.angular.z, self.yaw_scale)

        with self._rc_lock:
            self._rc = [lr, fb, ud, yaw]

    def _send_rc(self):
        """Send the latest RC command to Tello at 10 Hz."""
        with self._rc_lock:
            lr, fb, ud, yaw = self._rc
        try:
            self.tello.send_rc_control(lr, fb, ud, yaw)
        except Exception as e:
            self.get_logger().warn(f"send_rc_control failed: {e}", throttle_duration_sec=2.0)

    # ─── Services ─────────────────────────────────────────────────────────────

    def _takeoff_cb(self, _, response: Trigger.Response):
        try:
            self.tello.takeoff()
            response.success = True
            response.message = 'takeoff'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _land_cb(self, _, response: Trigger.Response):
        try:
            self.tello.land()
            response.success = True
            response.message = 'land'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _emergency_cb(self, _, response: Trigger.Response):
        try:
            self.tello.emergency()
            response.success = True
            response.message = 'emergency'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        try:
            self.tello.streamoff()
            self.tello.end()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelloInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()