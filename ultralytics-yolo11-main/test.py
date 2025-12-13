import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('D:/BaiduNetdiskDownload/yolo11魔鬼面具最新版/yolo11魔鬼面具最新版/ultralytics-yolo11-main/runs/train/exp45/weights/best.pt')   # 修改yaml
	#model.load('yolo11n.pt')  #加载预训练权重
	model.train(data='F:/yolov8/ultralytics-main/ultralytics-main/data/dataset/data.yaml',   #数据集yaml文件
	            imgsz=640,
	            epochs=200,
				patience=30, # 当验证集上的性能连续 30 轮次没有提升时触发早停[^2^]
	            batch=16,
	            workers=8,
	            device=0,   #没显卡则将0修改为'cpu'
	            optimizer='SGD',
                amp = False,
	            cache=False,   #服务器可设置为True，训练速度变快
	)
