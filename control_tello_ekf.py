"""
control_tello_ekf.py
====================
Pygame manual-control window for the Tello drone.
No ROS required — talks directly to the Tello via djitellopy.

Keybindings
-----------
  [T]         : Takeoff
  [L]         : Land
  [SPACE]     : Stop all motion
  [ESC]       : Emergency stop (cuts motors)
  W / S       : Up / Down
  A / D       : Yaw left / right
  8 / 5       : Forward / Backward
  4 / 6       : Left / Right
"""

import sys
import math
import threading

import numpy as np
import pygame


class TelloControlWindow:
    """
    Pygame window that reads keyboard input and sends RC commands to the Tello.

    Parameters
    ----------
    tello : djitellopy.Tello
        Connected, stream-enabled Tello object.
    ekf : EKFLocalization | None
        Optional EKF object used only for display.
    speed_pct : int
        Default speed percentage (0-100).
    yaw_pct : int
        Default yaw percentage (0-100).
    """

    def __init__(self, tello, ekf=None, speed_pct: int = 30, yaw_pct: int = 40):
        self.tello     = tello
        self.ekf       = ekf
        self.speed_pct = speed_pct
        self.yaw_pct   = yaw_pct

        # RC state (percentages -100 to 100)
        self.lr  = 0
        self.fb  = 0
        self.ud  = 0
        self.yaw = 0

        pygame.init()
        self.screen = pygame.display.set_mode((500, 460))
        pygame.display.set_caption('Tello EKF Controller')
        self.font_l = pygame.font.SysFont('monospace', 22, bold=True)
        self.font_m = pygame.font.SysFont('monospace', 19)
        self.font_s = pygame.font.SysFont('monospace', 15)

        self._running = True

    def run(self):
        """Blocking main loop. Returns when window is closed."""
        clock = pygame.time.Clock()
        while self._running:
            self._handle_events()
            self._send_rc()
            self._render()
            clock.tick(20)
        self.tello.send_rc_control(0, 0, 0, 0)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                self._on_key_down(event.key)
            elif event.type == pygame.KEYUP:
                self._on_key_up(event.key)

    def _on_key_down(self, key):
        if key == pygame.K_t:
            threading.Thread(target=self.tello.takeoff, daemon=True).start()
        elif key == pygame.K_l:
            threading.Thread(target=self.tello.land, daemon=True).start()
        elif key == pygame.K_ESCAPE:
            self.tello.emergency()
        elif key == pygame.K_SPACE:
            self.lr = self.fb = self.ud = self.yaw = 0
        elif key == pygame.K_w:                         self.ud  =  self.speed_pct
        elif key == pygame.K_s:                         self.ud  = -self.speed_pct
        elif key == pygame.K_a:                         self.yaw = -self.yaw_pct
        elif key == pygame.K_d:                         self.yaw =  self.yaw_pct
        elif key in (pygame.K_8, pygame.K_KP8):         self.fb  =  self.speed_pct
        elif key in (pygame.K_5, pygame.K_KP5):         self.fb  = -self.speed_pct
        elif key in (pygame.K_4, pygame.K_KP4):         self.lr  = -self.speed_pct
        elif key in (pygame.K_6, pygame.K_KP6):         self.lr  =  self.speed_pct

    def _on_key_up(self, key):
        if   key in (pygame.K_w, pygame.K_s):                                      self.ud  = 0
        elif key in (pygame.K_a, pygame.K_d):                                      self.yaw = 0
        elif key in (pygame.K_8, pygame.K_KP8, pygame.K_5, pygame.K_KP5):         self.fb  = 0
        elif key in (pygame.K_4, pygame.K_KP4, pygame.K_6, pygame.K_KP6):         self.lr  = 0

    def _send_rc(self):
        try:
            self.tello.send_rc_control(self.lr, self.fb, self.ud, self.yaw)
        except Exception as e:
            print(f"[Control] RC error: {e}")

    def _render(self):
        BG    = (28,  30,  42)
        WHITE = (220, 222, 235)
        CYAN  = ( 80, 210, 255)
        GREEN = ( 90, 220, 120)
        YLW   = (255, 210,  60)
        GRAY  = (120, 122, 140)
        RED   = (255,  80,  80)

        self.screen.fill(BG)
        y = 14

        def line(text, color=WHITE, font=None):
            nonlocal y
            surf = (font or self.font_m).render(text, True, color)
            self.screen.blit(surf, (14, y))
            y += surf.get_height() + 4

        line("  Tello EKF Controller", WHITE, self.font_l)
        line("-" * 42, GRAY, self.font_s)
        try:
            bat = self.tello.get_battery()
            h   = self.tello.get_height()
            bc  = GREEN if bat > 30 else (YLW if bat > 15 else RED)
            line(f"  Battery: {bat:3d}%   Height: {h} cm", bc)
        except Exception:
            line("  Battery: ---   Height: ---", GRAY)

        if self.ekf is not None and self.ekf.is_initialized:
            p = self.ekf.pose
            line(f"  EKF  x={p[0]:.2f}  y={p[1]:.2f}  z={p[2]:.2f}", CYAN)
            line(f"       yaw={math.degrees(p[4]):.1f} deg", CYAN)
        else:
            line("  EKF: waiting for first AprilTag...", GRAY)

        line("-" * 42, GRAY, self.font_s)
        line("  [T] Takeoff   [L] Land   [ESC] Emergency", WHITE, self.font_s)
        line("  [SPACE] Stop all", WHITE, self.font_s)
        line("-" * 42, GRAY, self.font_s)
        line(f"  [W/S] Up/Down      ud  = {self.ud:+4d}%")
        line(f"  [A/D] Yaw L/R      yaw = {self.yaw:+4d}%")
        line(f"  [8/5] Fwd/Back     fb  = {self.fb:+4d}%")
        line(f"  [4/6] Left/Right   lr  = {self.lr:+4d}%")
        pygame.display.flip()

    def stop(self):
        self._running = False
