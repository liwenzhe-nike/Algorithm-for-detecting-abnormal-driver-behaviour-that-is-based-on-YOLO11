import os
import cv2
import random
import numpy as np

# 配置
DATASET_DIR = r"./driver_abnormal_behavior_dataset"  # 数据集根目录
ID_TO_CLASS = {
    0: "Phone",
    1: "drinking",
    2: "yawning",
    3: "seatblt",
    4: "drowsy",
    5: "smoking"
}
CLASS_COLOR = {  # 每个类别对应不同颜色（BGR格式）
    0: (0, 255, 0),    # Phone - 绿色
    1: (255, 0, 0),    # drinking - 蓝色
    2: (0, 0, 255),    # yawning - 红色
    3: (255, 255, 0),  # seatblt - 青色
    4: (255, 0, 255),  # drowsy - 品红
    5: (0, 255, 255)   # smoking - 黄色
}

def draw_yolo_label(img, label_path):
    """在图片上绘制YOLO标签的边界框和类别"""
    img_h, img_w = img.shape[:2]
    # 读取标签文件
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # 转换为像素坐标（左上角和右下角）
        x1 = int((x - w/2) * img_w)
        y1 = int((y - h/2) * img_h)
        x2 = int((x + w/2) * img_w)
        y2 = int((y + h/2) * img_h)
        # 绘制边界框
        color = CLASS_COLOR.get(class_id, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 绘制类别名称
        class_name = ID_TO_CLASS.get(class_id, "Unknown")
        cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def view_random_sample(subset="train"):
    """随机查看一个样本（train/val/test）"""
    # 构建路径
    img_dir = os.path.join(DATASET_DIR, subset, "img")
    label_dir = os.path.join(DATASET_DIR, subset, "label")
    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        print(f"{subset}集没有图片！")
        return
    # 随机选一张
    img_name = random.choice(img_files)
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print("读取图片失败！")
        return
    # 绘制标签
    img_with_label = draw_yolo_label(img.copy(), label_path)
    # 调整窗口大小（方便查看）
    img_with_label = cv2.resize(img_with_label, (800, 600))
    # 显示图片
    cv2.imshow(f"Sample: {subset}/{img_name}", img_with_label)
    print(f"查看的样本：{subset}/{img_name}")
    print(f"标签文件：{label_path}")
    # 按任意键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 可选：train/val/test
    view_random_sample(subset="train")
    # view_random_sample(subset="val")
    # view_random_sample(subset="test")