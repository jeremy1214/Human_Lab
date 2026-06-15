import os
import time
import subprocess
import random
import cv2
import numpy as np
import onnxruntime as ort

import Start_Tello
import Balloon_Detector

# --- Classification constants (match image_classifier_node) ---
CLASS_NAMES = ['cap', 'brr', 'trala', 'tung']
LANDING_TAG = {'cap': 13, 'brr': 14, 'trala': 15, 'tung': 16}
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESH = 0.15
YOLO_NMS_THRESH = 0.45
CLASSIFY_FRAMES = 15
MIN_VOTE_FRACTION = 0.5


def _letterbox(img: np.ndarray, new_size: int = 640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad_y, pad_x = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized
    return canvas, scale, pad_x, pad_y


def preprocess_yolo(frame: np.ndarray):
    lb, scale, pad_x, pad_y = _letterbox(frame, YOLO_INPUT_SIZE)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_x, pad_y


def postprocess_yolo(output: np.ndarray, conf_thresh: float = YOLO_CONF_THRESH, nms_thresh: float = YOLO_NMS_THRESH):
    raw = output[0]
    rows = raw.T
    boxes_xywh = rows[:, :4]
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]
    mask = confidences >= conf_thresh
    if not np.any(mask):
        return []
    boxes_f = boxes_xywh[mask]
    confs_f = confidences[mask]
    class_f = class_ids[mask]
    x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
    y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
    x2 = boxes_f[:, 0] + boxes_f[:, 2] / 2
    y2 = boxes_f[:, 1] + boxes_f[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    detections = []
    for cid in np.unique(class_f):
        idx_c = np.where(class_f == cid)[0]
        bxs_c = boxes_xyxy[idx_c].tolist()
        cfs_c = confs_f[idx_c].tolist()
        keep = cv2.dnn.NMSBoxes(
            [[x, y, x2 - x, y2 - y] for x, y, x2, y2 in bxs_c],
            cfs_c, conf_thresh, nms_thresh
        )
        if len(keep) == 0:
            continue
        for k in (keep.flatten() if isinstance(keep, np.ndarray) else keep):
            i = idx_c[k]
            cls_id = int(class_f[i])
            detections.append({
                'class_id': cls_id,
                'class_name': CLASS_NAMES[cls_id],
                'confidence': float(confs_f[idx_c.tolist().index(i)]),
                'box': [float(v) for v in bxs_c[k]]
            })
    return sorted(detections, key=lambda d: d['confidence'], reverse=True)


def find_brainrot_model(filename: str = 'brainrot_detect.onnx') -> str:
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), 'Apriltag', 'model', filename),
        os.path.join(os.path.dirname(__file__), 'Apriltag', 'tello_localization', filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find '{filename}' in candidates: {candidates}")


def main():
    # 1) Initialize Tello (Stage 1)
    print("[Final Game] Initializing Tello (Stage 1)...")
    tello = Start_Tello.initialize_tello_stage1()

    # Draw random starting angle for fairness (back-facing to object zone)
    drawn_angle = random.choice([0, 90, 180, 270])
    Start_Tello.rotate_to_start_angle(tello, target_yaw=drawn_angle)
    print(f'[Final Game] Drawn starting angle: {drawn_angle} degrees')

    # 2) Run balloon detection / tracking (Stage 2 & 3)
    print("[Final Game] Starting balloon detection & tracking (Stage 2/3)...")
    # load the ONNX model used by Balloon_Detector
    try:
        balloon_net = cv2.dnn.readNetFromONNX("balloon.onnx")
    except Exception as e:
        print(f"[Final Game] Warning: failed to load balloon.onnx: {e}")
        balloon_net = None

    # initialize Kalman filter
    kf = Balloon_Detector.init_kalman_filter()
    kf_initialized = False
    lost_counter = 0
    pid_states = (0, 0, 0, 0, 0, 0)

    print('[Final Game] Entering detection loop. Press Ctrl-C to abort.')
    try:
        while True:
            frame = tello.get_frame_read().frame
            if frame is None:
                time.sleep(0.01)
                continue

            box = None
            if balloon_net is not None:
                box = Balloon_Detector.detect_balloon_onnx(frame, balloon_net)

            if box is not None:
                print('[Final Game] Balloon detected — engaging tracking/controller')
                # show detection on screen for scoring (Chinese message)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
                cv2.putText(frame, '有偵測到balloon', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.imshow('HCC Final Integration', frame)
                lost_counter = 0
                h, w = frame.shape[:2]
                z_meas = Balloon_Detector.recover_3d_position(box, w, h)

                if not kf_initialized:
                    kf.statePost[0:3] = z_meas
                    kf.statePost[3:6] = 0
                    kf_initialized = True
                else:
                    kf.correct(z_meas)

                tracked_pos = kf.statePost[0:3]

                # call controller helper from Balloon_Detector
                is_touched, pid_states = Balloon_Detector.track_and_control_tello(tello, tracked_pos, pid_states)
                if is_touched:
                    print('[Final Game] Touch action completed — proceeding to image classification')
                    break
            else:
                # continue searching (small yaw rotation)
                Balloon_Detector.search_balloon_pattern(tello, search_speed=25)
                cv2.imshow('HCC Final Integration', frame)

            # small sleep so loop is not busy
            time.sleep(0.02)

    except KeyboardInterrupt:
        print('[Final Game] Aborted by user — landing Tello and exiting.')
        tello.land()
        tello.streamoff()
        return

    # stop motion and stream before moving to next stage
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

    # 3) Perform image classification (voting) using onboard ONNX model
    print('[Final Game] Capturing frames for image classification (Stage 4)...')

    def classify_frames(tello, model_path: str):
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        votes = []
        conf_sum = 0.0
        captured = 0
        window = 'Classification'
        cv2.namedWindow(window)
        while captured < CLASSIFY_FRAMES:
            frame = tello.get_frame_read().frame
            if frame is None:
                time.sleep(0.01)
                continue
            blob, scale, pad_x, pad_y = preprocess_yolo(frame)
            output = sess.run(None, {input_name: blob})[0]
            dets = postprocess_yolo(output)
            label = None
            conf = 0.0
            if dets:
                label = dets[0]['class_name']
                conf = dets[0]['confidence']
                votes.append(label)
                conf_sum += conf
                cv2.putText(frame, f"Detected: {label} {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(frame, "No detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            captured += 1
        cv2.destroyWindow(window)
        if not votes:
            return None, 0.0
        # decide by majority
        from collections import Counter
        counts = Counter(votes)
        label, cnt = counts.most_common(1)[0]
        fraction = cnt / len(votes)
        avg_conf = conf_sum / len(votes) if len(votes) > 0 else 0.0
        if fraction >= MIN_VOTE_FRACTION:
            return label, avg_conf
        return None, avg_conf

    try:
        model_path = find_brainrot_model()
    except FileNotFoundError as e:
        print(f'[Final Game] Classification model not found: {e}')
        model_path = None

    class_label = None
    class_conf = 0.0
    if model_path is not None:
        try:
            class_label, class_conf = classify_frames(tello, model_path)
        except Exception as e:
            print(f'[Final Game] Classification failed: {e}')

    if class_label is None:
        print('[Final Game] Classification inconclusive — defaulting to cap')
        class_label = 'cap'
    print(f"[Final Game] Classification result: {class_label} (avg conf {class_conf:.2f})")

    # 3b) Launch image classifier node (Stage 4) inside same process (Option A)
    print('[Final Game] Starting image classifier node (Stage 4) inside this process...')
    try:
        import threading
        import rclpy
        from Apriltag.tello_localization.image_classifier_node import ImageClassifierNode, Stage

        rclpy.init()
        ic_node = ImageClassifierNode()
        # pass classification result into the node and start navigation
        ic_node.class_label = class_label
        ic_node.target_tag_id = LANDING_TAG.get(class_label)
        # move node state to navigating so it begins tag search
        try:
            ic_node.stage = Stage.NAVIGATING
        except Exception:
            pass

        def _spin_node():
            try:
                rclpy.spin(ic_node)
            except KeyboardInterrupt:
                pass
            finally:
                try:
                    ic_node.destroy_node()
                except Exception:
                    pass
                try:
                    rclpy.shutdown()
                except Exception:
                    pass

        th = threading.Thread(target=_spin_node, daemon=True)
        th.start()
        print(f'[Final Game] Image classifier node started in-thread (name={th.name}).')

        # Wait for the node thread to finish (or until interrupted)
        try:
            th.join()
        except KeyboardInterrupt:
            print('[Final Game] Interrupted; shutting down image classifier node...')
            try:
                rclpy.shutdown()
            except Exception:
                pass

    except Exception as e:
        # Fallback: spawn as subprocess if rclpy or import fails
        print(f"[Final Game] In-process start failed ({e}), falling back to subprocess.")
        script_path = os.path.join(os.path.dirname(__file__), 'Apriltag', 'tello_localization', 'image_classifier_node.py')
        if not os.path.exists(script_path):
            print(f"[Final Game] ERROR: cannot find image classifier script at: {script_path}")
        else:
            proc = subprocess.Popen(['python', script_path])
            print(f'[Final Game] Image classifier started (pid={proc.pid}). Attach to its terminal or logs to monitor.')
            try:
                proc.wait()
            except KeyboardInterrupt:
                print('[Final Game] KeyboardInterrupt, terminating image classifier subprocess...')
                proc.terminate()

    print('[Final Game] Done. Cleaning up Tello.')
    try:
        tello.land()
    except Exception:
        pass
    try:
        tello.streamoff()
    except Exception:
        pass


if __name__ == '__main__':
    main()
