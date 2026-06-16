import argparse
import os
import time

import cv2
import numpy as np
from djitellopy import Tello


def load_yolo_model(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    try:
        from ultralytics import YOLO

        return YOLO(weights_path)
    except Exception:
        import torch

        return torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)


def extract_detections(results):
    detections = []

    if results is None:
        return detections

    try:
        if hasattr(results, 'xyxy'):
            dets = results.xyxy[0]
            if dets is None:
                return detections
            dets = dets.cpu().numpy()
            for x1, y1, x2, y2, conf, cls in dets:
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
            return detections

        if isinstance(results, list) and len(results) > 0:
            first = results[0]
            if hasattr(first, 'boxes'):
                boxes = first.boxes
                if boxes is None:
                    return detections
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf[i]), int(cls[i])))
                return detections
    except Exception:
        pass

    return detections


def draw_detections(frame, detections, names, color):
    for x1, y1, x2, y2, conf, cls in detections:
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Tello camera detection for balloon and brainrot')
    parser.add_argument('--balloon', default='balloon.pt', help='Path to balloon detection model weights')
    parser.add_argument('--brainrot', default='brainrot_detect.pt', help='Path to brainrot detection model weights')
    parser.add_argument('--width', type=int, default=960, help='Display width')
    parser.add_argument('--height', type=int, default=720, help='Display height')
    args = parser.parse_args()

    print('Connecting to Tello...')
    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_reader = tello.get_frame_read()

    print('Loading models...')
    balloon_model = load_yolo_model(args.balloon)
    brainrot_model = load_yolo_model(args.brainrot)

    balloon_names = getattr(balloon_model, 'names', None)
    brainrot_names = getattr(brainrot_model, 'names', None)
    if balloon_names is None:
        balloon_names = {}
    if brainrot_names is None:
        brainrot_names = {}

    window_name = 'Tello Balloon + Brainrot Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (args.width, args.height))

            try:
                balloon_results = balloon_model(frame)
                brainrot_results = brainrot_model(frame)
            except Exception as ex:
                print('Detection error:', ex)
                time.sleep(0.1)
                continue

            balloon_dets = extract_detections(balloon_results)
            brainrot_dets = extract_detections(brainrot_results)

            annotated = frame.copy()
            annotated = draw_detections(annotated, balloon_dets, balloon_names, (0, 255, 0))
            annotated = draw_detections(annotated, brainrot_dets, brainrot_names, (0, 0, 255))

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                filename = f'tello_capture_{int(time.time())}.jpg'
                cv2.imwrite(filename, annotated)
                print(f'Saved {filename}')
    finally:
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
