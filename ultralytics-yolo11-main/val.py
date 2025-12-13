
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r'D:/BaiduNetdiskDownload/yolo11train/yolo11魔鬼面具最新版/ultralytics-yolo11-main/runs/train/exp121/weights/best.pt') # 选择训练好的权重路径
    model.val(data='F:/yolov8/ultralytics-main/ultralytics-main/data/dataset/data.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=4,
              workers=4,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )