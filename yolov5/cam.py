import cv2
import torch
from pathlib import Path

# 載入模型
model = torch.hub.load('yolov5', 'custom', path=r'C:\Users\ab881\OneDrive\桌面\test\YOLOV5\training_result\content\yolov5\runs\train\exp2\weights\best.pt', source='local')

# 設定鏡頭 (0 代表內建鏡頭)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 鏡頭打不開")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 讀取鏡頭失敗")
        break

    # 推論
    results = model(frame)

    # 繪圖
    results.render()  # 直接畫在 frame 上
    frame = results.ims[0]  # 取得畫好框的影像 (numpy array)

    # 顯示
    cv2.imshow("YOLOv5 Camera Detection", frame)

    # 按 'q' 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()