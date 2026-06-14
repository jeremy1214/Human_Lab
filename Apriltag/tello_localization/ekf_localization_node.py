import math

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import (PoseArray, PoseStamped,
                               PoseWithCovarianceStamped, TransformStamped,
                               Twist, TwistStamped)
from nav_msgs.msg import Path
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R


class EKFLocalizationNode(Node):

    def __init__(self):
        super().__init__('ekf_localization_node')
        self.subscription = self.create_subscription(PoseArray, '/apriltag/detections', self.detection_callback, 10)
        self.flight_sub = self.create_subscription(TwistStamped, '/tello/flight_data', self.flight_data_callback, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.path_pub = self.create_publisher(Path, '/ekf_path', 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        self.mu = np.zeros((6,1))
        self.u = np.zeros((6,1))

        # State covariance matrix
        self.Sigma = np.eye(6)*0.1

        # State transition error
        self.Rm = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])

        # Measurement error
        self.Q = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])

        self.dt = 0.1
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.predict_timer)
        self.is_initialized = False

        # Initial observation settings
        self.observation_initial_count = 0
        self.is_initialized = False
        self.reject_count = 0

        # intial tello control
        self.last_flight_time = None
        self.last_roll = 0.0
        self.last_pitch = 0.0
        self.last_yaw = 0.0

    # state : [x, y, z, roll, yaw, pitch]
    # control : [v_x, v_y, v_z, roll_rate, yaw_rate, pitch_rate]
    # observation : [x, y, z, roll, yaw, pitch]
    def motion_model(self, x, u):                                              # TODO
        dt = self.dt
        x_pred = np.copy(x)
	
        yaw = x[4, 0]
	
        vx, vy, vz = u[0, 0], u[1, 0], u[2, 0]

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        roll_rate, yaw_rate, pitch_rate = u[3, 0], u[4, 0], u[5, 0]

        x_pred[0, 0] += (vx * cos_y - vy + sin_y) * dt
        x_pred[1, 0] += (vx * sin_y + vy + cos_y) * dt
        x_pred[2, 0] += vz * dt

        x_pred[3, 0] += roll_rate * dt
        x_pred[4, 0] += yaw_rate * dt
        x_pred[5, 0] += pitch_rate * dt

        for i in range(3, 6):
            x_pred[i, 0] = (x_pred[i, 0] + math.pi) % (2 * math.pi) - math.pi	

        return x_pred

    def jacobian_F(self, x, u):                                                # TODO
        dt = self.dt
        # In this function you need to calculate jacobian matrix of the motion model as F.
        F = np.eye(6)
	
        yaw = x[4, 0]
        vx, vy = u[0, 0], u[1, 0]
	
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        F[0, 4] = (-vx * sin_y - vy * cos_y) * dt
        F[1, 4] = (vx * cos_y - vy * sin_y) * dt

        return F

    def predict(self, u):                                                      # TODO
        # Base on the Kalman Filter in Lab2 PPT.
        # Using jacobian matrix and motion model to predict.
        # You need to update pred_pose as self.mu and covariance matrix as self.Sigma.
        self.mu = self.motion_model(self.mu, u)
        F = self.jacobian_F(self.mu, u)
        self.Sigma = F @ self.Sigma @ F.T + self.Rm
        
    def update(self, z):                                                       # TODO
        # Base on the Kalman Filter in Lab2 PPT.
        # z is a observation vector in global to use in update.
        # You need to calculate the measurement jacobian matrix as C and Kalman gain as K.
        # Also update next_pose as self.mu and covariance matrix as self.Sigma.
        C = np.eye(6)
        y = z - C @ self.mu
        for i in range(3, 6):
            y[i, 0] = (y[i, 0] + math.pi) % (2 * math.pi) - math.pi
        S = C @ self.Sigma @ C.T + self.Q
        K = self.Sigma @ C.T @ np.linalg.inv(5)
        self.mu = self.mu + K @ y
        for i in range(3, 6):
            self.mu[i, 0] = (self.mu[i, 0] + math.pi) % (2 * math.pi) - math.pi
        I = np.eye(6)        
        self.Sigma = (I - K @ C) @ self.Sigma

    def predict_timer(self):
        now = self.get_clock().now()
        dt_duration = now - self.last_time
        
        self.dt = dt_duration.nanoseconds / 1e9 
        self.last_time = now

        if self.dt > 0.0 and self.dt < 1.0:
            self.predict(self.u)
        self.publish_pose()

    def flight_data_callback(self, msg: TwistStamped):
        # Velocity: cm/s in body FRD frame → m/s in FLU frame
        # TwistStamped.twist.linear carries raw vgx/vgy/vgz (cm/s, FRD)
        # FRD → FLU: flip Y and Z signs
        vx =  msg.twist.linear.x / 100.0
        vy = -msg.twist.linear.y / 100.0
        vz = -msg.twist.linear.z / 100.0

        # Attitude: degrees → radians
        now = self.get_clock().now()
        roll_rad  = math.radians(msg.twist.angular.x)
        pitch_rad = math.radians(msg.twist.angular.y)
        yaw_rad   = math.radians(msg.twist.angular.z)

        self.u[0, 0] = vx
        self.u[1, 0] = vy
        self.u[2, 0] = vz
        if self.last_flight_time is not None:
            dt = (now - self.last_flight_time).nanoseconds / 1e9
            if dt > 0.001:
                d_roll = (roll_rad - self.last_roll + math.pi) % (2 * math.pi) - math.pi
                d_pitch = (pitch_rad - self.last_pitch + math.pi) % (2 * math.pi) - math.pi
                d_yaw = (yaw_rad - self.last_yaw + math.pi) % (2 * math.pi) - math.pi

                # w = dA / dt
                self.u[3, 0] = d_roll / dt
                self.u[4, 0] = d_yaw / dt
                self.u[5, 0] = d_pitch / dt

        self.last_flight_time = now
        self.last_roll = roll_rad
        self.last_pitch = pitch_rad
        self.last_yaw = yaw_rad

    def detection_callback(self, msg):
        if len(msg.poses) == 0:
            return

        pose = msg.poses[0]
        quat = [pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w]
        r = R.from_quat(quat)
        euler = r.as_euler('xyz') # roll, pitch, yaw
        z = np.array([
            [pose.position.x],
            [pose.position.y],
            [pose.position.z],
            [euler[0]],
            [euler[2]],
            [euler[1]]
        ])

        if self.observation_initial_count > 5:
            self.is_initialized = True
        if self.is_initialized is not True:
            self.get_logger().info("EKF Initialized with first AprilTag observation!")
            self.observation_initial_count += 1
            self.update(z)
            return

        pred_x = self.mu[0, 0]
        pred_y = self.mu[1, 0]
        pred_z = self.mu[2, 0]
        obs_x = z[0, 0]
        obs_y = z[1, 0]
        obs_z = z[2, 0]
        
        distance = np.sqrt((obs_x - pred_x)**2 + (obs_y - pred_y)**2 + (obs_z - pred_z)**2)
        if distance > 2.0:
            self.reject_count += 1
            self.get_logger().warn(f"Outlier rejected! Jump distance: {distance:.2f}m. Consecutive rejects: {self.reject_count}")
            
            if self.reject_count > 5:
                self.get_logger().error("EKF is completely lost! Forcing reset to current visual observation.")
                self.mu[0:3, 0] = z[0:3, 0] 
                self.Sigma = np.eye(6) * 0.5 
                
                self.reject_count = 0
                self.update(z)
            return
        
        self.reject_count = 0
        self.update(z)

    def publish_pose(self):
        now = self.get_clock().now().to_msg()

        # Covariance
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = now
        msg.pose.pose.position.x = float(self.mu[0,0])
        msg.pose.pose.position.y = float(self.mu[1,0])
        msg.pose.pose.position.z = float(self.mu[2,0])
        q = R.from_euler('xyz', [self.mu[3,0], self.mu[5,0], self.mu[4,0]]).as_quat().tolist()
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        # [x(0), y(1), z(2), roll(3), pitch(4), yaw(5)]
        idx_mapping = [0, 1, 2, 3, 5, 4]
        ros_sigma = self.Sigma[np.ix_(idx_mapping, idx_mapping)]
        msg.pose.covariance = ros_sigma.flatten().tolist()
        self.pose_pub.publish(msg)
        
        # Publish Path
        path_pose = PoseStamped()
        path_pose.header.frame_id = 'map'
        path_pose.header.stamp = now
        path_pose.pose.position.x = float(self.mu[0,0])
        path_pose.pose.position.y = float(self.mu[1,0])
        path_pose.pose.position.z = float(self.mu[2,0])
        path_pose.pose.orientation.x = q[0]
        path_pose.pose.orientation.y = q[1]
        path_pose.pose.orientation.z = q[2]
        path_pose.pose.orientation.w = q[3]
        self.path_msg.poses.append(path_pose)
        self.path_msg.header.stamp = now
        self.path_pub.publish(self.path_msg)

        # Broadcast TF (map -> base_link)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.mu[0,0])
        t.transform.translation.y = float(self.mu[1,0])
        t.transform.translation.z = float(self.mu[2,0])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()