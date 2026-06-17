import time
import cv2
from djitellopy import Tello

# 初始化並連線 Tello
tello = Tello()
tello.connect()
print(f"目前電量: {tello.get_battery()}%")

# 開啟視訊串流
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)  # 等待 2 秒確保串流視訊穩定

# 起飛
# tello.takeoff()

# 建立顯示畫面的視窗
window_name = "Tello Live Stream (Space: Capture, Q: Land)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("=== 飛行控制提示 ===")
print("1. 請點擊視訊視窗，確保焦點在畫面上")
print("2. 按下【空白鍵 (Space)】：拍一張照片並儲存")
print("3. 按下【Q 鍵】或【ESC】：結束拍照並自動降落")
print("====================")

try:
    while True:
        # 取得無人機當前的畫面
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.01)
            continue

        # 顯示畫面在視窗上
        cv2.imshow(window_name, frame)

        # 偵測鍵盤按鍵 (等待 1 毫秒)
        key = cv2.waitKey(1) & 0xFF

        # 1. 如果按下空白鍵 (ASCII 碼為 32)
        if key == 32:
            filename = f"picture/tello_capture_{int(time.time())}.png"
            # 拍照並儲存原圖
            cv2.imwrite(filename, frame)
            print(f"📸 拍照成功！照片已儲存至: {filename}")

        # 2. 如果按下 Q 鍵 (ASCII 碼為 113) 或 ESC 鍵 (27)
        elif key == ord('q') or key == 27:
            print("收到結束指令，準備降落...")
            break

finally:
    # 確保不論程式如何結束，都會安全安全降落並關閉視窗
    print("安全程序啟動：無人機降落中...")
    # tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    print("程式安全結束。")