"""
Simple test script to verify balloon.onnx runs under ONNX Runtime.
Usage:
  python tools/test_balloon_onnxruntime.py --model balloon.onnx --image path/to/image.jpg
If --image is omitted the script creates a synthetic image with a red circle.

Outputs:
 - prints ONNX Runtime input name and output shapes
 - if outputs follow YOLO-like layout, prints best detection box (x,y,w,h)
"""
import argparse
import os
import sys
import numpy as np
import cv2


def make_synthetic_image(w=640, h=480):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # draw a red circle roughly resembling a balloon
    cv2.circle(img, (w//2, h//2), min(w,h)//6, (0,0,255), -1)
    return img


def preprocess(img, size=(416,416)):
    img_r = cv2.resize(img, size)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    img_r = img_r.astype(np.float32) / 255.0
    inp = np.transpose(img_r, (2,0,1))[None, ...]
    return inp


def parse_outputs_for_best_box(outputs, orig_w, orig_h, conf_thresh=0.25):
    best_box = None
    best_conf = 0.0
    for out in outputs:
        arr = np.array(out)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 5:
            continue
        for det in arr:
            scores = det[5:]
            if scores.size == 0:
                continue
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf > conf_thresh and conf > best_conf:
                best_conf = conf
                cx, cy, bw, bh = (det[0:4] * np.array([orig_w, orig_h, orig_w, orig_h])).astype(int)
                best_box = (int(cx - bw//2), int(cy - bh//2), int(bw), int(bh))
    return best_box, best_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='balloon.onnx')
    parser.add_argument('--image', '-i', default=None)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        print('onnxruntime not available:', e)
        sys.exit(2)

    if not os.path.exists(args.model):
        print('Model file not found:', args.model)
        sys.exit(3)

    # load session
    try:
        sess = ort.InferenceSession(args.model)
    except Exception as e:
        print('Failed to create ONNX Runtime session:', e)
        sys.exit(4)

    # load image
    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        if img is None:
            print('Failed to read image:', args.image)
            sys.exit(5)
    else:
        print('No image provided or not found; using synthetic test image.')
        img = make_synthetic_image(640, 480)

    orig_h, orig_w = img.shape[:2]
    inp = preprocess(img, size=(416,416))

    # determine model input shape and adapt preprocessing if needed
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    model_shape = input_meta.shape  # e.g. [1,3,320,320] or [1,320,320,3]
    print('ONNX Runtime input name:', input_name, 'shape:', model_shape)

    # adapt input if model expects different spatial size or NHWC layout
    try:
        # infer H,W and layout
        if len(model_shape) == 4:
            if model_shape[1] == 3 or (isinstance(model_shape[1], str) and '3' in str(model_shape[1])):
                # NCHW
                _, c, h, w = model_shape
                target_size = (int(w) if w is not None else 416, int(h) if h is not None else 416)
                inp = preprocess(img, size=target_size)
            else:
                # NHWC
                _, h, w, c = model_shape
                target_size = (int(w) if w is not None else 416, int(h) if h is not None else 416)
                img_r = cv2.resize(img, target_size)
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                img_r = img_r.astype(np.float32) / 255.0
                inp = img_r[None, ...]
        else:
            inp = inp

        outputs = sess.run(None, {input_name: inp})
    except Exception as e:
        print('ONNX Runtime session.run failed:', e)
        sys.exit(6)

    print('Received', len(outputs), 'output arrays')
    for i, out in enumerate(outputs):
        a = np.array(out)
        print(f' - output[{i}] shape={a.shape} dtype={a.dtype}')

    # Use the project's Balloon_Detector high-level API to test end-to-end
    try:
        import Balloon_Detector as bd
    except Exception:
        # fallback: load by file path
        try:
            from importlib.machinery import SourceFileLoader
            # provide a minimal fake Start_Tello module to satisfy imports when loading
            import types
            fake_mod = types.ModuleType('Start_Tello')
            fake_mod.initialize_tello_stage1 = lambda: None
            fake_mod.rotate_to_start_angle = lambda x: None
            import sys as _sys
            _sys.modules['Start_Tello'] = fake_mod
            bd = SourceFileLoader('Balloon_Detector', os.path.join(os.getcwd(), 'Balloon_Detector.py')).load_module()
        except Exception as e:
            print('Failed to import Balloon_Detector module by path:', e)
            sys.exit(7)

    try:
        box = bd.detect_balloon(img, ort_sess=sess)
    except Exception as e:
        print('Balloon_Detector.detect_balloon raised:', e)
        sys.exit(8)

    if box is not None:
        print('Balloon_Detector returned box:', box)
        x,y,w_box,h_box = box
        img_vis = img.copy()
        cv2.rectangle(img_vis, (x,y), (x+w_box, y+h_box), (0,255,0), 2)
        out_path = 'tools/test_balloon_result.jpg'
        cv2.imwrite(out_path, img_vis)
        print('Wrote visualization to', out_path)
    else:
        print('Balloon_Detector returned no detection (None).')

if __name__ == '__main__':
    main()
