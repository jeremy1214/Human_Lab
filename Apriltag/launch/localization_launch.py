import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
     package_name = 'tello_localization'
     rviz_config = os.path.join(get_package_share_directory(package_name), 'rviz', 'config.rviz')

     # Optional: launch the Stage-4 image classifier node
     # Usage:  ros2 launch tello_localization localization_launch.py stage4:=true
     stage4_arg = DeclareLaunchArgument('stage4', default_value='false',
                                        description='Launch image_classifier_node for Stage 4')

     # ====================================================
     # Tello Interface  (djitellopy bridge — replaces tello_driver)
     # ====================================================
     # Publishes  : /image_raw, /tello/flight_data
     # Subscribes : /cmd_vel
     # Services   : /tello/takeoff, /tello/land, /tello/emergency
     tello_interface_node = Node(package=package_name,
                                 executable='tello_interface_node',
                                 name='tello_interface_node',
                                 output='screen')

     # ====================================================
     # AprilTag Detector
     # ====================================================
     apriltag_node = Node(package=package_name,
                          executable='apriltag_detector_node',
                          name='apriltag_detector_node',
                          output='screen')

     # ====================================================
     # Static Tag TF
     # ====================================================
     tag_tf_node = Node(package=package_name,
                        executable='tag_tf_broadcaster',
                        name='tag_tf_broadcaster',
                        output='screen')

     # ====================================================
     # EKF Localization
     # ====================================================
     ekf_node = Node(package=package_name,
                    executable='ekf_localization_node',
                    name='ekf_localization_node',
                    output='screen')

     # ====================================================
     # Tello Manual Control
     # ====================================================
     control_node = Node(package=package_name,
                         executable='control_tello_ekf',
                         name='control_tello_ekf',
                         output='screen')

     # ====================================================
     # Stage 4: Image Classifier & Precision Landing
     # ====================================================
     classifier_node = Node(package=package_name,
                            executable='image_classifier_node',
                            name='image_classifier_node',
                            output='screen',
                            condition=IfCondition(LaunchConfiguration('stage4')))

     # ====================================================
     # Bind camera_frame to base_link
     # ====================================================
     static_tf_node = Node(package='tf2_ros',
                           executable='static_transform_publisher',
                           name='camera_base_link_tf',
                           arguments=['0', '0', '0', '-1.5708', '0', '-1.5708', 'base_link', 'camera_frame'])

     # ====================================================
     # RViz
     # ====================================================
     rviz_node = Node(package='rviz2',
                      executable='rviz2',
                      name='rviz2',
                      output='screen',
                      arguments=['-d', rviz_config])

     return LaunchDescription([
         stage4_arg,
         tello_interface_node,    # ← djitellopy bridge (was tello_driver)
         apriltag_node,
         tag_tf_node,
         ekf_node,
         control_node,
         classifier_node,
         static_tf_node,
         rviz_node,
     ])