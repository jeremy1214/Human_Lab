"""
Start_Tello.py
==============
Stage 1 helper functions: connect, takeoff, rotate to the drawn angle.
Used by Final_Game.py — not meant to be run directly.
"""

import time

from djitellopy import Tello


def initialize_tello_stage1():
    """Connect to the Tello, start the video stream, and take off."""
    tello = Tello()
    tello.connect()
    print(f"Tello 電量: {tello.get_battery()}%")

    tello.streamon()
    time.sleep(2)   # let the video stream stabilise

    print("[Stage 1] 執行自主起飛...")
    tello.takeoff()
    time.sleep(1.5)   # let the hover settle
    return tello


def rotate_to_start_angle(tello, target_yaw):
    """
    Rotate to the drawn starting angle (back-facing the object zone).

    Parameters
    ----------
    tello : djitellopy.Tello
    target_yaw : int
        Degrees to rotate. Positive = clockwise, negative = counter-clockwise.
    """
    if target_yaw == 0:
        print("[Stage 1] 抽籤角度為 0 度，保持原方向。")
        return

    print(f"[Stage 1] 執行抽籤初始角度轉向: {target_yaw} 度")
    if target_yaw > 0:
        tello.rotate_clockwise(target_yaw)
    else:
        tello.rotate_counter_clockwise(abs(target_yaw))

    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)
    print("[Stage 1] 初始角度調整完畢，進入尋找目標狀態。")