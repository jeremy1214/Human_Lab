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
    try:
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
    except Exception as exc:
        print(f"[Error] 初始化 Tello 失敗: {exc}")
        try:
            tello.end()
        except Exception:
            pass
        raise


def rotate_to_start_angle(tello):
    """
    根據抽籤決定的角度進行初始轉向（通常是背對 object zone，增加比賽隨機性）。
    
    :param tello: Tello 飛行器物件
    :param target_yaw: 抽籤抽到的旋轉角度 (整數，正數為順時針，負數為逆時針)
    """
    tello.rotate_clockwise(180)
        
    # 旋轉後懸停 1 秒確保機身定點，準備切換至影像偵測
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)
    print("[Stage 1] 初始角度調整完畢，進入尋找目標狀態。")

def main():
    # 0. 讀取現場抽籤結果 (例如抽到背對 90 度)
    # 正式比賽時，只需修改此處輸入的變數即可
    drawn_angle = 90
    tello = None

    try:
        # 1. 呼叫 Stage 1 初始化與起飛函式
        tello = initialize_tello_stage1()

        # 2. 呼叫 Stage 1 抽籤旋轉函式
        rotate_to_start_angle(tello)

        print("[FSM] 開始競賽任務迴圈...")
        while True:
            frame = tello.get_frame_read().frame
            if frame is None:
                time.sleep(0.01)
                continue

            # 顯示影像畫面
            cv2.imshow("HCC Final Integration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[FSM] 收到退出指令 q，準備降落")
                try:
                    tello.land()
                except Exception as exc:
                    print(f"[Error] 著陸失敗: {exc}")
                break
    except Exception as exc:
        print(f"[Error] 執行期間發生例外: {exc}")
    finally:
        if tello is not None:
            try:
                tello.streamoff()
            except Exception as exc:
                print(f"[Warning] 影像串流關閉失敗: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"[Warning] 關閉視窗失敗: {exc}")

if __name__ == "__main__":
    main()