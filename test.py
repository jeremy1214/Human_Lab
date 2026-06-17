#!/usr/bin/env python3
"""
Use the Tello camera to test brainrot classification without flying.

This script connects to the Tello and starts the video stream only.
It never calls takeoff(), land(), go_xyz_speed(), rotate_*, or send_rc_control().
The classification logic intentionally matches Final_Game.py:
  - YOLO .pt model loaded with Ultralytics
  - model.predict(frame, conf=CLASSIFY_CONF_THRESH)
  - use results[0].boxes
  - take the highest-confidence box
  - majority vote over CLASSIFY_FRAMES detections

Run from the folder containing brainrot_detect.pt:
  python test_brainrot_tello_camera.py --model brainrot_detect.pt

Keys:
  q / ESC : quit
  s       : save current frame
"""

import argparse
import os
import time
from collections import Counter

import cv2
from djitellopy import Tello
from ultralytics import YOLO

CLASSIFY_FRAMES = 20
MIN_VOTE_FRACTION = 0.50
CLASSIFY_CONF_THRESH = 0.15

LANDING_TAG = {
    "cap": 13,
    "brr": 14,
    "trala": 15,
    "tung": 16,
}


def classify_image(frame, model, conf_thresh: float = CLASSIFY_CONF_THRESH):
    """
    Same classification function as Final_Game.py.
    Returns (label, confidence) or (None, 0.0).
    """
    if model is None:
        return None, 0.0

    results = model.predict(frame, verbose=False, conf=conf_thresh)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, 0.0

    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    conf = float(confs[best])
    if conf < conf_thresh:
        return None, 0.0

    cls_id = int(boxes.cls.cpu().numpy()[best])
    label = model.names[cls_id]
    return label, conf


def draw_overlay(frame, label, conf, votes, conf_sum, battery=None):
    display = frame.copy()

    if label is None:
        main_text = "No detection - hold image closer/steadier"
        color = (0, 80, 255)
    else:
        tag_id = LANDING_TAG.get(label, "?")
        main_text = f"{label} -> AprilTag {tag_id}  conf={conf:.2f}"
        color = (0, 255, 0) if tag_id != "?" else (0, 200, 255)

    cv2.putText(display, main_text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    if votes:
        counts = Counter(votes)
        best, count = counts.most_common(1)[0]
        fraction = count / len(votes)
        avg_conf = conf_sum / max(len(votes), 1)
        vote_text = (
            f"votes {len(votes)}/{CLASSIFY_FRAMES}: {dict(counts)}  "
            f"best={best} agreement={fraction:.0%} avg_conf={avg_conf:.2f}"
        )
        cv2.putText(display, vote_text, (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if len(votes) >= CLASSIFY_FRAMES:
            result_text = "ACCEPT" if fraction >= MIN_VOTE_FRACTION else "LOW AGREEMENT"
            cv2.putText(display, result_text, (10, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if result_text == "ACCEPT" else (0, 165, 255), 2)

    status = "Tello camera only - no takeoff commands"
    if battery is not None:
        status += f"   battery={battery}%"
    cv2.putText(display, status, (10, display.shape[0] - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(display, "q/ESC: quit   s: save frame",
                (10, display.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="brainrot_detect.pt",
                        help="Path to brainrot model, for example brainrot_detect.pt")
    parser.add_argument("--conf", type=float, default=CLASSIFY_CONF_THRESH,
                        help="Confidence threshold")
    parser.add_argument("--frames", type=int, default=CLASSIFY_FRAMES,
                        help="Number of detections to collect before printing a result")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print("[Init] Loading model...")
    model = YOLO(args.model)
    print("[Model names]", model.names)
    print("[Landing map]", LANDING_TAG)

    tello = Tello()
    votes = []
    conf_sum = 0.0

    try:
        print("[Init] Connecting to Tello Wi-Fi camera...")
        tello.connect()
        battery = tello.get_battery()
        print(f"[Tello] Battery: {battery}%")

        print("[Init] Starting video stream. No takeoff will be sent.")
        tello.streamon()
        time.sleep(2.0)
        frame_read = tello.get_frame_read()

        last_label = None
        last_conf = 0.0

        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.02)
                continue

            last_label, last_conf = classify_image(frame, model, args.conf)
            if last_label is not None:
                votes.append(last_label)
                conf_sum += last_conf

            if len(votes) >= args.frames:
                counts = Counter(votes)
                best, cnt = counts.most_common(1)[0]
                frac = cnt / len(votes)
                avg_conf = conf_sum / max(len(votes), 1)
                print("\n" + "=" * 48)
                print(f"  CLASSIFICATION RESULT : {best.upper()}")
                print(f"  Votes  : {dict(counts)}  agreement={frac:.0%}  avg_conf={avg_conf:.2f}")
                print(f"  TARGET : AprilTag {LANDING_TAG[best]}")
                print("=" * 48 + "\n")

                if frac >= MIN_VOTE_FRACTION:
                    votes = []
                    conf_sum = 0.0
                else:
                    print("[Classify] Low vote agreement - collecting more frames")
                    votes = []
                    conf_sum = 0.0

            display = draw_overlay(frame, last_label, last_conf, votes, conf_sum, battery)
            cv2.imshow("Tello Brainrot Classification Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                out = f"tello_brainrot_frame_{int(time.time())}.jpg"
                cv2.imwrite(out, frame)
                print(f"[Saved] {out}")

    finally:
        print("[Shutdown] Stopping stream.")
        try:
            tello.streamoff()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()