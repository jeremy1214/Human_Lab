import cv2
import math
import numpy as np
import time
from djitellopy import Tello
import Start_Tello

# Optional ONNX Runtime backend
try:
    import onnxruntime as ort
except Exception:
    ort = None

# =====================================================================
# 1. 全局參數與相機內參設定 (參考 State Estimation Lab 3)
# =====================================================================
# 相機內參 (from Lab 1 calibration)
FX, FY = 835.342103847164, 839.4691450667409
CX, CY = 415.5366635247159, 355.11975613817964

BALLOON_REAL_DIAMETER = 30.0  # 氣球實際直徑 (單位: 公分)，用於計算實體距離

# 追蹤 PID 增益設定 [Kp, Ki, Kd]
PID_X = [0.4, 0.0, 0.1]       # 左右誤差控制 (對應到 Tello 的 Yaw 軸自轉)
PID_Y = [0.4, 0.0, 0.1]       # 上下誤差控制 (對應到 Tello 的 Throttle 上下)
PID_Z = [0.5, 0.0, 0.1]       # 前後距離控制 (對應到 Tello 的 Pitch 前後)

# Flag: mark ONNX Runtime broken if inference fails
_ORT_BROKEN = False

def search_balloon_pattern(tello, search_speed=25):
    """ 自主巡航搜尋。當尚未看見氣球時，控制 Tello 原地緩慢自轉 """
    # 全程禁止手動操控，故使用內部指令給予固定 yaw 速度
    tello.send_rc_control(0, 0, 0, search_speed)

# =====================================================================
# 3. Stage 2/3 模組化功能函式 (Perception, Kalman & Control)
# =====================================================================

def init_kalman_filter():
    """ 初始化 Lab 3 的常速運動模型 (Constant Velocity Model) 卡爾曼濾波器 """
    # 狀態向量 x = [x, y, z, vx, vy, vz]^T (3D位置與3D速度)
    kf = cv2.KalmanFilter(6, 3, 0)
    kf.transitionMatrix = np.eye(6, dtype=np.float32) # dt 會在主迴圈動態更新
    
    # 量測矩陣 H (只能觀測到 3D 空間位置 x, y, z)
    kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1
    kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = 1
    
    # 濾波雜訊協方差設定 (依據 Lab 3 實測平滑度進行調校)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf

# Note: cv2.dnn ONNX path removed — ONNX Runtime (onnxruntime) is the sole model backend.


def detect_balloon_ort(frame, session):
    """Use onnxruntime InferenceSession to run the balloon model and return best box."""
    if session is None:
        return None
    h, w, _ = frame.shape
    # Determine model input layout/size from session
    try:
        input_meta = session.get_inputs()[0]
        model_shape = input_meta.shape
    except Exception:
        model_shape = None

    # Default preprocessing values
    target_size = (416, 416)
    use_nchw = True
    if model_shape and len(model_shape) == 4:
        # NCHW if second dim is 3
        if model_shape[1] == 3 or (isinstance(model_shape[1], str) and '3' in str(model_shape[1])):
            use_nchw = True
            _, c, mh, mw = model_shape
            target_size = (int(mw) if mw is not None else 416, int(mh) if mh is not None else 416)
        else:
            use_nchw = False
            _, mh, mw, c = model_shape
            target_size = (int(mw) if mw is not None else 416, int(mh) if mh is not None else 416)

    # Preprocess according to detected layout
    if use_nchw:
        img = cv2.resize(frame, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        inp = np.transpose(img, (2, 0, 1))[None, ...]
    else:
        img_r = cv2.resize(frame, target_size)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_r = img_r.astype(np.float32) / 255.0
        inp = img_r[None, ...]

    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})
    except Exception as exc:
        global _ORT_BROKEN
        _ORT_BROKEN = True
        print(f"[Balloon_Detector] ONNX Runtime inference failed: {exc}")
        return None

    if outputs is None:
        return None

    best_box = None
    max_conf = 0.0
    # outputs may be a list of arrays; iterate and handle expected shapes
    for output in outputs:
        out_arr = np.array(output)
        # handle (1,5,2100) -> [0] -> (5,2100) -> transpose -> (2100,5)
        if out_arr.ndim == 3 and out_arr.shape[0] == 1:
            out_arr = out_arr[0]

        if out_arr.ndim == 2 and out_arr.shape[0] == 5:
            # transpose to have rows = detections, cols = attributes
            out_arr = out_arr.T

        # now expect shape (N, M) where M >=5
        if out_arr.ndim != 2 or out_arr.shape[1] < 5:
            continue

        for detection in out_arr:
            # detection may be [cx, cy, w, h, conf] (single-class)
            if detection.size == 5:
                confidence = float(detection[4])
                if confidence > 0.5 and confidence > max_conf:
                    max_conf = confidence
                    cx, cy, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype(int)
                    x = int(cx - bw // 2)
                    y = int(cy - bh // 2)
                    # clip to image bounds
                    x = max(0, min(x, w-1))
                    y = max(0, min(y, h-1))
                    bw = max(1, min(bw, w - x))
                    bh = max(1, min(bh, h - y))
                    best_box = (x, y, int(bw), int(bh))
            else:
                # multi-class style: [x,y,w,h, conf?, cls_scores...]
                scores = detection[5:]
                if scores.size == 0:
                    continue
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > 0.5 and confidence > max_conf:
                    max_conf = confidence
                    cx, cy, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype(int)
                    x = int(cx - bw // 2)
                    y = int(cy - bh // 2)
                    x = max(0, min(x, w-1))
                    y = max(0, min(y, h-1))
                    bw = max(1, min(bw, w - x))
                    bh = max(1, min(bh, h - y))
                    best_box = (x, y, int(bw), int(bh))

    return best_box


def _detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]

    s_mask = cv2.inRange(hsv, (0, 80, 60), (180, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(s_mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 400:
        return None

    circ = 4 * math.pi * area / (cv2.arcLength(best, True) ** 2 + 1e-6)
    if circ < 0.35:
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    return (x, y, bw, bh)


def detect_balloon(frame, ort_sess=None):
    """Use ONNX Runtime session if available; otherwise fallback to color-based detection."""
    if ort_sess is not None and ort is not None and not _ORT_BROKEN:
        box = detect_balloon_ort(frame, ort_sess)
        if box is not None:
            return box
    return _detect_color(frame)


def recover_3d_position(bbox, img_w, img_h):
    """ 參考 Lab 3.2: 依據相機內參與氣球大小，將 2D 影像特徵還原為 3D 相對物理坐標 """
    bx, by, bw, bh = bbox
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    
    # 計算相對於主點的像素誤差
    u_err = cx - CX
    v_err = CY - cy  # 轉為向上為正
    
    # 相似三角形還原實體距離 Z (單位: 公分)
    z_distance = (BALLOON_REAL_DIAMETER * FX) / (float(bw) + 1e-5)
    
    # 反投影計算實體空間中的 X 與 Y 偏差
    x_distance = (u_err * z_distance) / FX
    y_distance = (v_err * z_distance) / FY
    
    return np.array([[x_distance], [y_distance], [z_distance]], dtype=np.float32)

def run_pid_core(error, prev_error, integral, pid_gains):
    """ 通用 PID 核心計算機 """
    kp, ki, kd = pid_gains
    integral += error
    derivative = error - prev_error
    output = (kp * error) + (ki * integral) + (kd * derivative)
    return int(np.clip(output, -100, 100)), error, integral

def track_and_control_tello(tello, tracked_pos, pid_states):
    """ 
    依據卡爾曼濾波後得到的平滑 3D 座標，計算三軸 PID 速度並發送給 Tello。
    當距離靠得夠近時，觸發「衝刺碰撞」指令。
    """
    x_p, y_p, z_p = tracked_pos[0][0], tracked_pos[1][0], tracked_pos[2][0]
    
    # 取出過去的 PID 狀態變數
    err_x_p, int_x, err_y_p, int_y, err_z_p, int_z = pid_states
    
    # 左右誤差 (x_p) 控制 Yaw 軸自轉對準，比直接側移平移更穩定，且不易丟失視野
    yaw_speed, err_x_p, int_x = run_pid_core(x_p, err_x_p, int_x, PID_X)
    ud_speed, err_y_p, int_y = run_pid_core(y_p, err_y_p, int_y, PID_Y)
    
    # 前後距離目標定在氣球前方 35 公分處，保留衝刺緩衝
    fb_speed, err_z_p, int_z = run_pid_core(z_p - 35, err_z_p, int_z, PID_Z)
    
    # 保存更新後的 PID 狀態
    updated_pid_states = (err_x_p, int_x, err_y_p, int_y, err_z_p, int_z)
    
    # 判定是否執行最後衝刺 (卡爾曼估計距離 <= 45cm 且對準中心)
    print("z_p:", z_p)
    if z_p <= 45.0 and abs(x_p) < 10 and not z_p==0:
        print("[Action] 進入終點線！執行最後向前衝刺碰撞！")
        tello.send_rc_control(0, 60, 0, 0)  # 直線全力加速向前
        time.sleep(0.8)
        tello.send_rc_control(0, 0, 0, 0)  # 碰撞後立即急煞懸停
        return True, updated_pid_states    # 回傳 True 代表碰撞完成
    
    # 正常追蹤控制發送
    tello.send_rc_control(0, int(np.clip(fb_speed, -40, 40)), int(np.clip(ud_speed, -30, 30)), int(np.clip(yaw_speed, -30, 30)))
    return False, updated_pid_states

# =====================================================================
# 4. 主程式狀態機統合 (Main FSM Loop)
# =====================================================================

def main():
    
    # 執行 Stage 1 初始化與定量轉向
    tello = Start_Tello.initialize_tello_stage1()
    Start_Tello.rotate_to_start_angle(tello)
    
    # 載入氣球偵測 ONNX 模型 (onnxruntime only)
    balloon_sess = None
    if ort is not None:
        try:
            balloon_sess = ort.InferenceSession("balloon.onnx")
            print("[Balloon_Detector] Loaded balloon.onnx with ONNX Runtime.")
        except Exception as exc:
            print(f"[Balloon_Detector] Failed to load ONNX Runtime session: {exc}")
            balloon_sess = None
    else:
        print("[Balloon_Detector] onnxruntime is not installed; balloon detection will use color fallback.")
    
    # 初始化卡爾曼濾波器與追蹤控制狀態
    kf = init_kalman_filter()
    kf_initialized = False
    lost_counter = 0
    
    # 初始化 PID 狀態紀錄 (err_x, int_x, err_y, int_y, err_z, int_z)
    pid_states = (0, 0, 0, 0, 0, 0)
    
    stage = "SEARCH_BALLOON"
    last_time = time.time()
    
    print("[FSM] 開始全自主任務監控...")
    while True:
        frame = tello.get_frame_read().frame
        if frame is None:
            continue
            
        frame_display = frame.copy()
        h, w, _ = frame.shape
        
        # 動態計算時間步長 dt，更新卡爾曼狀態轉移
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        kf.transitionMatrix[0, 3] = dt
        kf.transitionMatrix[1, 4] = dt
        kf.transitionMatrix[2, 5] = dt
        
        # 卡爾曼濾波器常速預測
        prediction = kf.predict()
        
        # 進行 YOLO 氣球感測 (onnxruntime -> color fallback)
        box = detect_balloon(frame, ort_sess=balloon_sess)
        
        # -------------------------------------------------------------
        # 有限狀態機核心邏輯
        # -------------------------------------------------------------
        if stage == "SEARCH_BALLOON":
            if box is not None:
                print("【得分 CP1】透過 Terminal 顯示：偵測到氣球！")
                tello.send_rc_control(0, 0, 0, 0) # 煞車
                stage = "TRACK_AND_TOUCH"
            else:
                # 尚未偵測到氣球，呼叫 Stage 1 模組進行自主旋轉搜尋
                search_balloon_pattern(tello, search_speed=25)
                
        elif stage == "TRACK_AND_TOUCH":
            tracked_pos = None
            
            if box is not None:
                lost_counter = 0
                cv2.rectangle(frame_display, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                
                # 呼叫 3D 還原函式
                z_meas = recover_3d_position(box, w, h)
                
                if not kf_initialized:
                    kf.statePost[0:3] = z_meas
                    kf.statePost[3:6] = 0
                    kf_initialized = True
                else:
                    # 卡爾曼更新 (Measurement Update)
                    kf.correct(z_meas)
                tracked_pos = kf.statePost[0:3]
            else:
                lost_counter += 1
                # Lab 3 特性：短暫丟失目標時，信任卡爾曼預測值繼續導引前進
                if kf_initialized and lost_counter < 15:
                    tracked_pos = prediction[0:3]
                    print("[Kalman Filter] 氣球短暫遺失，使用預測軌跡中...")
                else:
                    # 徹底跟丟，重設濾波器並切回 Stage 1 搜尋模式
                    print("[FSM] 徹底跟丟目標，切回搜尋階段。")
                    kf_initialized = False
                    tello.send_rc_control(0, 0, 0, 0)
                    stage = "SEARCH_BALLOON"
            
            # 執行追蹤控制
            if tracked_pos is not None:
                # 在畫面上繪製 Kalman 濾波後的平滑 3D 物理距離
                cv2.putText(frame_display, f"X: {tracked_pos[0][0]:.1f} Y: {tracked_pos[1][0]:.1f} Z: {tracked_pos[2][0]:.1f} cm", 
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # 呼叫追蹤控制與碰撞模組
                is_touched, pid_states = track_and_control_tello(tello, tracked_pos, pid_states)
                
                if is_touched:
                    print("【待評分 CP2】請助教判定是否有成功碰撞氣球。")
                    stage = "IMAGE_CLASSIFICATION"
                    # 比賽計時暫停點 (Stage 1~3 結束)
                    
        elif stage == "IMAGE_CLASSIFICATION":
            # 碰撞氣球成功，此處接續 Stage 4 迷因分類與 Stage 5 AprilTag 地圖降落
            tello.send_rc_control(0, 0, 0, 0)
            print("[FSM] 進入迷因圖分類與降落階段...")
            break # 範例在此處跳出

        cv2.imshow("HCC Final Stage 1 & 2 Integration", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()