#!/usr/bin/env python3
"""
test_vision_only.py — HCC 2026 視覺與定位純測試腳本 (不起飛、不開電機)
===========================================================================
"""

#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
from djitellopy import Tello
os.add_dll_directory("C:/Users/jerem/miniconda3/envs/hcc_env/Lib/site-packages/pupil_apriltags/lib/apriltag.dll")
# 載入自定義的 AprilTag 偵測器
from apriltag_detector import AprilTagDetector


# C:\Users\jerem\miniconda3\envs\hcc_env\Lib\site-packages\pupil_apriltags\lib\apriltag.dll
# 確保 ONNXRuntime 載入
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("請先安裝 onnxruntime: pip install onnxruntime")


# =====================================================================
# 1. 相機內參設定 (來自實驗室校正檔)
# =====================================================================
FX, FY = 835.342103847164, 839.4691450667409
CX, CY = 415.5366635247159, 355.11975613817964
BALLOON_REAL_DIAMETER = 30.0  # 氣球實體直徑 (cm)

# =====================================================================
# 2. ONNX 輔助函式
# =====================================================================
def load_onnx_model_ort(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到 ONNX 模型檔案: {weights_path}")
    print(f"[Init] 成功載入 ONNX 權重: {weights_path}")
    return ort.InferenceSession(weights_path, providers=['CPUExecutionProvider'])

def extract_ort_detections(session_outputs, img_w, img_h, conf_threshold=0.6):
    detections = []
    if not session_outputs:
        return detections
    output = session_outputs[0]
    if output.shape[1] < output.shape[2]:
        output = np.transpose(output, (0, 2, 1))
    predictions = output[0]

    for prediction in predictions:
        scores = prediction[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            cx, cy, bw, bh = prediction[0:4]
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            detections.append((max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2), float(confidence), int(class_id)))
    return detections

def main():
    print('正在連接 Tello (僅建立影像串流)...')
    tello = Tello()
    tello.connect()
    print(f"目前電量: {tello.get_battery()}%")
    
    # 開啟串流，但不呼叫 takeoff()
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2.0)

    # 載入氣球 ONNX 模型與 AprilTag 地圖
    # 請確保這兩個檔案跟此腳本在同一個資料夾下
    balloon_model_path = "balloon.onnx" 
    map_yaml_path = "apriltag_map.yaml"

    balloon_session = load_onnx_model_ort(balloon_model_path)
    balloon_input_name = balloon_session.get_inputs()[0].name
    tag_detector = AprilTagDetector(map_yaml_path)

    window_name = 'HCC Vision Test Window (ESC to Exit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n>>> 視覺測試啟動！您可以將 Tello 拿在手上移動進行測試。<<<")
    print(">>> 按下 [ESC] 鍵可關閉視窗並結束程式。 <<<\n")

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                continue

            frame_display = frame.copy()
            h, w, _ = frame.shape

            # ── 1. 測試 AprilTag 偵測與 3D Pose 解算 ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = tag_detector._detector.detect(gray, estimate_tag_pose=True, 
                                                 camera_params=(FX, FY, CX, CY), 
                                                 tag_size=0.165)
            
            if len(tags) > 0:
                for tag in tags:
                    # 標註 Tag 四個角點
                    for corner in tag.corners:
                        cv2.circle(frame_display, (int(corner[0]), int(corner[1])), 5, (0, 255, 255), -1)
                    
                    # 提取相對相機位移 (單位: 公尺)
                    t_x = tag.pose_t[0][0]
                    t_y = tag.pose_t[1][0]
                    t_z = tag.pose_t[2][0]
                    
                    text = f"Tag ID {tag.tag_id} -> X:{t_x:.2f}m, Y:{t_y:.2f}m, Z:{t_z:.2f}m"
                    cv2.putText(frame_display, text, (20, 40 + tag.tag_id * 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(frame_display, "No AprilTag Detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ── 2. 測試 YOLO ONNX 氣球偵測 ──
            img_in = cv2.resize(frame, (640, 640))
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32) / 255.0
            img_in = np.expand_dims(img_in, axis=0)

            balloon_outputs = balloon_session.run(None, {balloon_input_name: img_in})
            balloon_dets = extract_ort_detections(balloon_outputs, w, h, conf_threshold=0.5)

            if len(balloon_dets) > 0:
                best_box = balloon_dets[0]
                # 畫出氣球綠色辨識框
                cv2.rectangle(frame_display, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
                
                # 計算單點測距 (單位: 公分)
                bw = best_box[2] - best_box[0]
                z_distance = (BALLOON_REAL_DIAMETER * FX) / (float(bw) + 1e-5)
                
                cv2.putText(frame_display, f"Balloon Conf: {best_box[4]:.2f} Distance: {z_distance:.1f}cm", 
                            (best_box[0], best_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame_display, "No Balloon Detected", (20, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 顯示綜合畫面
            cv2.imshow(window_name, frame_display)
            
            # 按 ESC 退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        print("[System] 測試安全結束。")

if __name__ == '__main__':
    main()