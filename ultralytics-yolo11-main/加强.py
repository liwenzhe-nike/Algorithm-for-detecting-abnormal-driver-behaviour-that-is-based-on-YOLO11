import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
from ultralytics import YOLO

# ========= 可改参数 =========
WEIGHT      = 'D:/BaiduNetdiskDownload/yolo11train/yolo11魔鬼面具最新版/ultralytics-yolo11-main/runs/train/exp116/weights/best.pt'          # 你的模型
SRC_IMG_DIR = 'D:/BaiduNetdiskDownload/yolo数据集/J-驾驶行为检测13000YOLO/train/images'             # 原始大图文件夹
DST_ROOT    = 'screened_dataset'       # 输出根目录
CONF_THRES  = 0.35                     # 置信度阈值
TARGET_IDS  = {0,1,2,3,4,5}            # 模型里对应 6 类的 id
VISUALIZE   = True                     # 是否同时保存可视化图
# ============================

DST_IMG_DIR = Path(DST_ROOT) / 'images'
DST_LBL_DIR = Path(DST_ROOT) / 'labels'
DST_VIS_DIR = Path(DST_ROOT) / 'vis'
for p in (DST_IMG_DIR, DST_LBL_DIR, DST_VIS_DIR):
    p.mkdir(parents=True, exist_ok=True)

model = YOLO(WEIGHT)

img_paths = list(Path(SRC_IMG_DIR).rglob('*.*'))
for img_path in tqdm(img_paths, desc='Screening'):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    results = model(img, verbose=False)[0]          # 单张推理
    boxes   = results.boxes
    if boxes is None:
        continue
    # 过滤：置信度 + 类别
    keep = []
    for b in boxes:
        cls, conf = int(b.cls), float(b.conf)
        if cls in TARGET_IDS and conf >= CONF_THRES:
            keep.append(b)
    if not keep:
        continue

    # 1. 复制图片
    dst_img_path = DST_IMG_DIR / img_path.name
    shutil.copy2(img_path, dst_img_path)

    # 2. 写 yolo 格式 label
    h0, w0 = img.shape[:2]
    label_path = (DST_LBL_DIR / img_path.stem).with_suffix('.txt')
    with open(label_path, 'w') as f:
        for b in keep:
            cls = int(b.cls)
            x,y,w,h = b.xywhn[0].tolist()   # 已归一化
            f.write(f'{cls} {x} {y} {w} {h}\n')

    # 3. 可视化（可选）
    if VISUALIZE:
        vis = results.plot()                # ultralytics 自带画框
        cv2.imwrite(str(DST_VIS_DIR / img_path.name), vis)