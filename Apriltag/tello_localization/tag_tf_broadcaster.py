import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class TagTFBroadcaster(Node):
    def __init__(self):
        super().__init__('tag_tf_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        
        package_share = get_package_share_directory('tello_localization')
        yaml_path = os.path.join(package_share, 'map', 'apriltag_map.yaml')
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        transforms = []
        for tag in data['tags']:
            tag_id = tag['id']
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = f'tag_{tag_id}'
            t.transform.translation.x = float(tag['position'][0])
            t.transform.translation.y = float(tag['position'][1])
            t.transform.translation.z = float(tag['position'][2])

            q = R.from_euler('xyz', tag['orientation_rpy']).as_quat()
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            transforms.append(t)

        self.broadcaster.sendTransform(transforms)
        self.get_logger().info('Published AprilTag static TFs')


class RobotTFBroadcaster:
    def __init__(self, node):
        self.node = node
        self.br = TransformBroadcaster(node)

    def send(self, mu):
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(mu[0])
        t.transform.translation.y = float(mu[1])
        t.transform.translation.z = float(mu[2])

        q = R.from_euler('xyz', [mu[3], mu[5], mu[4]]).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = TagTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
