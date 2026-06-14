import cv2
import time
from djitellopy import Tello

# =====================================================================
# Stage 1: 模組化功能函式 (Functions)
# =====================================================================

def initialize_tello_stage1():
    """
    初始化 Tello 並執行安全起飛。
    """
    tello = Tello()
    tello.connect()
    print(f"Tello 電量: {tello.get_battery()}%")
    
    tello.streamon()
    # 給予影像串流建立的時間
    time.sleep(2) 
    
    print("[Stage 1] 執行自主起飛...")
    tello.takeoff()
    # 起飛後稍微懸停穩定機身
    time.sleep(1.5) 
    return tello


def rotate_to_start_angle(tello, target_yaw):
    """
    根據抽籤決定的角度進行初始轉向（通常是背對 object zone，增加比賽隨機性）。
    
    :param tello: Tello 飛行器物件
    :param target_yaw: 抽籤抽到的旋轉角度 (整數，正數為順時針，負數為逆時針)
    """
    if target_yaw == 0:
        print("[Stage 1] 抽籤角度為 0 度，保持原方向。")
        return

    print(f"[Stage 1] 執行抽籤初始角度轉向: {target_yaw} 度")
    
    if target_yaw > 0:
        tello.rotate_clockwise(target_yaw)
    else:
        tello.rotate_counter_clockwise(abs(target_yaw))
        
    # 旋轉後懸停 1 秒確保機身定點，準備切換至影像偵測
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)
    print("[Stage 1] 初始角度調整完畢，進入尋找目標狀態。")


def search_balloon_pattern(tello, search_speed=20):
    """
    自主搜尋控制指令。當 Stage 2 尚未偵測到氣球時，持續呼叫此函式讓 Tello 安全自轉。
    符合競賽「全程禁止手動操控鍵盤」的規範。
    
    :param tello: Tello 飛行器物件
    :param search_speed: 自轉速度 (deg/s)
    """
    # 僅給予 yaw 軸轉動速度，其餘平移軸皆為 0，確保原地定點旋轉
    tello.send_rc_control(0, 0, 0, search_speed)

def main():
    # 0. 讀取現場抽籤結果 (例如抽到背對 90 度)
    # 正式比賽時，只需修改此處輸入的變數即可
    drawn_angle = 90 
    
    # 1. 呼叫 Stage 1 初始化與起飛函式
    tello = initialize_tello_stage1()
    
    # 2. 呼叫 Stage 1 抽籤旋轉函式
    rotate_to_start_angle(tello, target_yaw=drawn_angle)
    
    # 3. 進入主狀態機迴圈
    stage = "SEARCH_BALLOON"
    balloon_net = cv2.dnn.readNetFromONNX("balloon.onnx")
    
    print("[FSM] 開始競賽任務迴圈...")
    while True:
        frame = tello.get_frame_read().frame
        if frame is None:
            continue
            
        # =============================================================
        # 統合後的狀態機邏輯
        # =============================================================
        if stage == "SEARCH_BALLOON":
            # 呼叫 Stage 2 的 YOLO 偵測功能
            box = detect_balloon_onnx(frame, balloon_net)
            
            if box is not None:
                print("【得分 CP1】透過 Terminal 顯示：偵測到氣球！")
                # 發現目標，立即停止自轉並切換至卡爾曼追蹤階段 (Stage 2 / 3)
                tello.send_rc_control(0, 0, 0, 0)
                stage = "TRACK_AND_TOUCH"
            else:
                # 沒偵測到氣球，持續套用 Stage 1 的自主旋轉搜尋模組
                search_balloon_pattern(tello, search_speed=25)
                
        elif stage == "TRACK_AND_TOUCH":
            # 這裡直接跑你寫好的卡爾曼濾波更新與 PID 控制代碼...
            pass
            
        # 顯示影像畫面
        cv2.imshow("HCC Final Integration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()