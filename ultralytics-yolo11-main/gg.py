from ultralytics import YOLO
import cv2
import torch
from pathlib import Path

def detect_camera():
    # 模型配置
    model_config = {
        'model_path': r'yolo11n.pt',  # 本地模型路径，注意配置
        'download_url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'  # 如果没有模型文件下载URL
    }

    # 推理参数
    predict_config = {
        'conf_thres': 0.25,     # 置信度阈值
        'iou_thres': 0.45,      # IoU阈值
        'imgsz': 640,           # 输入分辨率
        'line_width': 2,        # 检测框线宽
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    }

    # 加载模型（带异常捕获）
    if not Path(model_config['model_path']).exists():
        if model_config['download_url']:
            print("开始下载模型...")
            YOLO(model_config['download_url']).download(model_config['model_path'])
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_config['model_path']}")

    # 初始化模型
    model = YOLO(model_config['model_path']).to(predict_config['device'])
    print(f"✅ 模型加载成功 | 设备: {predict_config['device'].upper()}")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 实时检测循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 执行推理
        results = model.predict(
            source=frame,
            stream=True,  # 流式推理
            verbose=False,
            conf=predict_config['conf_thres'],
            iou=predict_config['iou_thres'],
            imgsz=predict_config['imgsz'],
            device=predict_config['device']
        )

        # 遍历生成器获取结果（取第一个结果）
        for result in results:
            annotated_frame = result.plot(line_width=predict_config['line_width'])
            break

        # 显示实时画面
        cv2.imshow('YOLO Real-time Detection', annotated_frame)

        # 按键退出q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 检测结束")

if __name__ == "__main__":
    detect_camera()