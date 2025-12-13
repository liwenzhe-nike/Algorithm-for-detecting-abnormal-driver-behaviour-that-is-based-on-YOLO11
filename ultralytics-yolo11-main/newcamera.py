# -*- coding: utf-8 -*-
"""
YOLOv11 实时摄像头检测 + 警报逻辑 + 窗口最大化
python >=3.7, 依赖: ultralytics, opencv-python
"""

from ultralytics import YOLO
import cv2

# ========================= 1. 模型/摄像头初始化 =========================
model_path = r'D:/BaiduNetdiskDownload/yolo11train/yolo11魔鬼面具最新版/ultralytics-yolo11-main/runs/train/exp113/weights/best.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)          # 0=默认摄像头
if not cap.isOpened():
    print('❌ 无法打开摄像头')
    exit()

# ========================= 2. 警报参数 =========================
ALERT_CLS      = [2, 4]   # 需要警报的类别
ALERT_THRES    = 20       # 连续多少帧触发警报
alert_counter  = 0        # 连续检测到警报类别的帧数

# ========================= 3. 主循环 =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print('❌ 无法读取帧')
        break

    # 3.1 推理
    results = model(frame, verbose=False)   # verbose=False 关掉 ultralytics 日志

    # 3.2 解析结果
    alert_flag = False
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'

            # 画框 + 标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 警报类别额外提示
            if cls in ALERT_CLS:
                alert_flag = True
                cv2.putText(frame, 'dangerous', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 3.3 警报逻辑
    if alert_flag:
        alert_counter += 1
        if alert_counter >= ALERT_THRES:
            # 大红框
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (50, 50), (w - 50, h - 50), (0, 0, 255), 10)
            cv2.putText(frame, 'DANGROUS', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    else:
        alert_counter = 0

    # 3.4 显示（窗口最大化）
    cv2.namedWindow('YOLO Real-time Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('YOLO Real-time Detection',
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow('YOLO Real-time Detection', frame)

    # 3.5 退出键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========================= 4. 清理 =========================
cap.release()
cv2.destroyAllWindows()