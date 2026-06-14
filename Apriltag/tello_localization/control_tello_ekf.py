#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction  # publish tello takeoff & land
import pygame
import sys

class ControlTelloEKF(Node):
    def __init__(self):
        super().__init__('control_tello_ekf')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.client_ = self.create_client(TelloAction, '/tello_action')
        
        pygame.init()
        self.screen = pygame.display.set_mode((480, 420))
        pygame.display.set_caption('Tello & EKF Controller (GTA 5 Mode)')
        self.font = pygame.font.SysFont(None, 24)
        
        # initial control signals (6 DoF)
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.roll_rate  = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate   = 0.0

        self.speed_step = 0.1
        self.angle_step = 0.3

        # scanning in 20Hz
        self.timer = self.create_timer(0.05, self.timer_callback)

    def send_tello_cmd(self, cmd_string):
        if not self.client_.service_is_ready():
            self.get_logger().warn(f"Tello Action Service can not publish cmd: {cmd_string}")
            return
        request = TelloAction.Request()
        request.cmd = cmd_string
        self.client_.call_async(request)
        self.get_logger().info(f"Publish cmd: {cmd_string}")

    def timer_callback(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.KEYDOWN:
                # tello action mod change (takeoff & land)
                if event.key == pygame.K_t: 
                    self.send_tello_cmd('takeoff')
                elif event.key == pygame.K_l: 
                    self.send_tello_cmd('land')

                # Stop function keys
                if event.key == pygame.K_SPACE: 
                    self.v_x = self.v_y = self.v_z = 0.0
                    self.roll_rate = self.pitch_rate = self.yaw_rate = 0.0

                # Left hand (WASD)
                elif event.key == pygame.K_w: self.v_z = self.speed_step
                elif event.key == pygame.K_s: self.v_z = -self.speed_step
                elif event.key == pygame.K_a: self.yaw_rate = self.angle_step
                elif event.key == pygame.K_d: self.yaw_rate = -self.angle_step
                
                # Right hand (8456)
                elif event.key in [pygame.K_KP8, pygame.K_8]: 
                    self.v_x = self.speed_step
                    self.pitch_rate = self.angle_step
                elif event.key in [pygame.K_KP5, pygame.K_5]: 
                    self.v_x = -self.speed_step
                    self.pitch_rate = -self.angle_step
                elif event.key in [pygame.K_KP4, pygame.K_4]: 
                    self.v_y = self.speed_step
                    self.roll_rate = self.angle_step
                elif event.key in [pygame.K_KP6, pygame.K_6]: 
                    self.v_y = -self.speed_step
                    self.roll_rate = -self.angle_step
                    
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s]: self.v_z = 0.0
                elif event.key in [pygame.K_a, pygame.K_d]: self.yaw_rate = 0.0
                elif event.key in [pygame.K_KP8, pygame.K_8, pygame.K_KP5, pygame.K_5]: 
                    self.v_x = 0.0
                    self.pitch_rate = 0.0
                elif event.key in [pygame.K_KP4, pygame.K_4, pygame.K_KP6, pygame.K_6]: 
                    self.v_y = 0.0
                    self.roll_rate = 0.0

        # --- 4. update Pygame window ---
        self.screen.fill((40, 44, 52))
        info_text = [
            " [ Tello EKF Controller : ]",
            " * Keep this window focused to control *",
            " ---------------------------------------",
            "  [T] : TAKEOFF  /  [L] : LAND ",
            " ---------------------------------------",
            " --- Left Hand (WASD) ---",
            f"   v_z (w/s) : {self.v_z:.2f}",
            f"   yaw_rate (a/d) : {self.yaw_rate:.2f}",
            " --- Right Hand (8/4/5/6) ---",
            f"   v_x & pitch_rate (8/5) : {self.v_x:.2f} , {self.pitch_rate:.2f}",
            f"   v_y & roll_rate (4/6) : {self.v_y:.2f} , {self.roll_rate:.2f}",
            "",
            " Press SPACE to STOP ALL"
        ]
        
        y_offset = 15
        for line in info_text:
            text_surface = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(text_surface, (20, y_offset))
            y_offset += 28
        pygame.display.flip()


        velocity_print = f"\rinput control = [vx:{self.v_x:.2f}, vy:{self.v_y:.2f}, vz:{self.v_z:.2f}, "
        ryp_rate_print = f"roll:{self.roll_rate:.2f}, pitch:{self.pitch_rate:.2f}, yaw:{self.yaw_rate:.2f}]"
        sys.stdout.write(velocity_print + ryp_rate_print)
        sys.stdout.flush()

        # publish control to Tello
        twist = Twist()
        twist.linear.x = float(self.v_x)
        twist.linear.y = float(self.v_y)
        twist.linear.z = float(self.v_z)
        twist.angular.x = float(self.roll_rate)
        twist.angular.y = float(self.pitch_rate)
        twist.angular.z = float(self.yaw_rate)
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ControlTelloEKF()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.publisher_.publish(twist)
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
