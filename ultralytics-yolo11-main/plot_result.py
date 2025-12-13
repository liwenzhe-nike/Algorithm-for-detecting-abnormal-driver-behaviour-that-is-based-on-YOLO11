import os
import random
import pandas as pd
from shutil import copy2
from collections import defaultdict
import cv2
import numpy as np

# -------------------------- 核心配置 --------------------------
ORIGINAL_DATA_DIR = r"E:/BaiduNetdiskDownload/YOLO"
SPLIT_DATA_DIR = r"./driver_abnormal_behavior_dataset"
AUG_DATA_DIR = r"./augmented_dataset"  # 增强后的临时目录
# 目标数量
TARGET_TRAIN_NUM = 2099
TARGET_VAL_NUM = 598
TARGET_TEST_NUM = 300
# 标签ID → 行为名称映射
ID_TO_CLASS = {
    0: "Phone",
    1: "drinking",
    2: "yawning",
    3: "seatblt",
    4: "drowsy",
    5: "smoking"
}
# 行为名称 → 预期数量（需增强到的数量）
CLASS_TARGET = {
    "Phone": 521,
    "drinking": 535,
    "yawning": 487,
    "seatblt": 685,  # 实际仅81张，需增强604张
    "drowsy": 511,
    "smoking": 503
}
# 文件后缀
IMG_SUFFIX = ('.jpg', '.jpeg', '.png', '.bmp')
LABEL_SUFFIX = '.txt'
# 随机种子
RANDOM_SEED = 42

# -------------------------- 初始化随机种子 --------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------- 工具函数 --------------------------
def read_label_file(label_path):
    """读取标签文件，返回类别ID和所有边界框（x,y,w,h）"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            with open(label_path, 'r', encoding=enc) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                bboxes = []
                class_id = None
                for line in lines:
                    parts = line.split()
                    parts = [p for p in parts if p]
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    bboxes.append((x, y, w, h))
                    if class_id is None:
                        class_id = cid
                return class_id, bboxes
        except Exception:
            continue
    return None, []


def yolo_to_pixel(bbox, img_w, img_h):
    """将YOLO格式（x,y,w,h 归一化）转换为像素坐标（x1,y1,x2,y2）"""
    x, y, w, h = bbox
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2


def pixel_to_yolo(bbox, img_w, img_h):
    """将像素坐标（x1,y1,x2,y2）转换为YOLO格式（x,y,w,h 归一化）"""
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2 / img_w
    y = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x, y, w, h


def augment_hflip(img, bboxes):
    """水平翻转图片和标签"""
    img_hflip = cv2.flip(img, 1)
    img_w = img.shape[1]
    bboxes_hflip = []
    for bbox in bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, img_w, img.shape[0])
        # 水平翻转后x坐标变换：x1' = img_w - x2, x2' = img_w - x1
        x1_new = img_w - x2
        x2_new = img_w - x1
        bbox_yolo = pixel_to_yolo((x1_new, y1, x2_new, y2), img_w, img.shape[0])
        bboxes_hflip.append(bbox_yolo)
    return img_hflip, bboxes_hflip


def augment_vflip(img, bboxes):
    """垂直翻转图片和标签"""
    img_vflip = cv2.flip(img, 0)
    img_h = img.shape[0]
    bboxes_vflip = []
    for bbox in bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, img.shape[1], img_h)
        # 垂直翻转后y坐标变换：y1' = img_h - y2, y2' = img_h - y1
        y1_new = img_h - y2
        y2_new = img_h - y1
        bbox_yolo = pixel_to_yolo((x1, y1_new, x2, y2_new), img.shape[1], img_h)
        bboxes_vflip.append(bbox_yolo)
    return img_vflip, bboxes_vflip


def augment_rotate(img, bboxes, angle_range=(-10, 10)):
    """随机旋转图片和标签（小角度）"""
    angle = random.uniform(angle_range[0], angle_range[1])
    img_h, img_w = img.shape[:2]
    # 计算旋转矩阵
    center = (img_w / 2, img_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 旋转图片
    img_rot = cv2.warpAffine(img, M, (img_w, img_h))
    # 旋转边界框
    bboxes_rot = []
    for bbox in bboxes:
        x1, y1, x2, y2 = yolo_to_pixel(bbox, img_w, img_h)
        # 转换四个角点
        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        points_rot = cv2.transform(points.reshape(-1, 1, 2), M).reshape(-1, 2)
        # 计算新的边界框
        x1_new = max(0, min(points_rot[:, 0]))
        y1_new = max(0, min(points_rot[:, 1]))
        x2_new = min(img_w, max(points_rot[:, 0]))
        y2_new = min(img_h, max(points_rot[:, 1]))
        # 过滤无效边界框
        if x2_new - x1_new < 1 or y2_new - y1_new < 1:
            continue
        bbox_yolo = pixel_to_yolo((x1_new, y1_new, x2_new, y2_new), img_w, img_h)
        bboxes_rot.append(bbox_yolo)
    return img_rot, bboxes_rot


def augment_brightness(img, bboxes, brightness_range=(0.7, 1.3)):
    """调整亮度"""
    brightness = random.uniform(brightness_range[0], brightness_range[1])
    img_bright = np.clip(img * brightness, 0, 255).astype(np.uint8)
    return img_bright, bboxes  # 亮度调整不改变标签


def augment_noise(img, bboxes, noise_scale=(0, 0.05 * 255)):
    """添加高斯噪声"""
    noise = np.random.normal(0, random.uniform(noise_scale[0], noise_scale[1]), img.shape)
    img_noise = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img_noise, bboxes  # 噪声添加不改变标签


def augment_image_and_label(img_path, label_path, aug_type):
    """
    增强图片和标签（纯OpenCV实现）
    aug_type: 增强类型（hflip/vflip/rotate/brightness/noise）
    返回：增强后的图片数组、增强后的边界框列表
    """
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    # 读取标签
    _, bboxes = read_label_file(label_path)
    if not bboxes:
        return None, []

    # 执行增强
    if aug_type == 'hflip':
        img_aug, bboxes_aug = augment_hflip(img, bboxes)
    elif aug_type == 'vflip':
        img_aug, bboxes_aug = augment_vflip(img, bboxes)
    elif aug_type == 'rotate':
        img_aug, bboxes_aug = augment_rotate(img, bboxes)
    elif aug_type == 'brightness':
        img_aug, bboxes_aug = augment_brightness(img, bboxes)
    elif aug_type == 'noise':
        img_aug, bboxes_aug = augment_noise(img, bboxes)
    else:
        return img, bboxes

    return img_aug, bboxes_aug


def save_augmented_file(img_aug, bboxes_aug, class_id, img_path, save_dir, aug_suffix):
    """保存增强后的图片和标签文件"""
    # 构建保存路径
    img_name = os.path.basename(img_path)
    name_prefix = os.path.splitext(img_name)[0]
    aug_img_name = f"{name_prefix}_{aug_suffix}{os.path.splitext(img_name)[1]}"
    aug_label_name = f"{name_prefix}_{aug_suffix}{LABEL_SUFFIX}"
    img_dst = os.path.join(save_dir, "img", aug_img_name)
    label_dst = os.path.join(save_dir, "label", aug_label_name)

    # 保存图片
    cv2.imwrite(img_dst, img_aug)
    # 保存标签
    with open(label_dst, 'w', encoding='utf-8') as f:
        for x, y, w, h in bboxes_aug:
            f.write(f"{class_id} {x} {y} {w} {h}\n")

    return img_dst, label_dst


# -------------------------- 第一步：扫描原始样本 --------------------------
sample_info = defaultdict(list)  # cls_name → [(img_path, label_path), ...]
print("===== 扫描原始数据集 =====")

for root, dirs, files in os.walk(ORIGINAL_DATA_DIR):
    if os.path.basename(root).lower() != "images":
        continue
    # 找到对应的labels文件夹
    parent_dir = os.path.dirname(root)
    label_dir = os.path.join(parent_dir, "labels")
    if not os.path.exists(label_dir):
        for d in os.listdir(parent_dir):
            if d.lower() == "labels":
                label_dir = os.path.join(parent_dir, d)
                break
    if not os.path.exists(label_dir):
        continue

    # 扫描图片
    for file in files:
        if file.lower().endswith(IMG_SUFFIX):
            img_path = os.path.join(root, file)
            sample_name = os.path.splitext(file)[0]
            label_path = os.path.join(label_dir, sample_name + LABEL_SUFFIX)
            if not os.path.exists(label_path):
                continue
            # 读取类别ID
            class_id, _ = read_label_file(label_path)
            if class_id not in ID_TO_CLASS:
                continue
            cls_name = ID_TO_CLASS[class_id]
            sample_info[cls_name].append((img_path, label_path))

# 输出原始样本数
for cls, samples in sample_info.items():
    print(f"   {cls}: {len(samples)} 张")

# -------------------------- 第二步：数据增强 --------------------------
# 创建增强目录
os.makedirs(os.path.join(AUG_DATA_DIR, "img"), exist_ok=True)
os.makedirs(os.path.join(AUG_DATA_DIR, "label"), exist_ok=True)

aug_types = ['hflip', 'vflip', 'rotate', 'brightness', 'noise']
aug_sample_info = defaultdict(list)  # 增强后的样本信息

print("\n===== 开始数据增强 =====")
for cls_name, samples in sample_info.items():
    target_num = CLASS_TARGET[cls_name]
    current_num = len(samples)
    need_aug_num = target_num - current_num

    if need_aug_num <= 0:
        aug_sample_info[cls_name] = samples
        print(f"   {cls_name}：原始样本数{current_num}≥目标数{target_num}，无需增强")
        continue

    print(f"   {cls_name}：需增强{need_aug_num}张（原始{current_num}张→目标{target_num}张）")
    aug_count = 0
    while aug_count < need_aug_num:
        # 随机选择一个原始样本
        img_path, label_path = random.choice(samples)
        # 随机选择增强类型
        aug_type = random.choice(aug_types)
        # 增强图片和标签
        img_aug, bboxes_aug = augment_image_and_label(img_path, label_path, aug_type)
        if img_aug is None or len(bboxes_aug) == 0:
            continue
        # 保存增强后的文件
        aug_suffix = f"{aug_type}_{aug_count}"
        class_id = [k for k, v in ID_TO_CLASS.items() if v == cls_name][0]
        img_dst, label_dst = save_augmented_file(
            img_aug, bboxes_aug, class_id, img_path, AUG_DATA_DIR, aug_suffix
        )
        aug_sample_info[cls_name].append((img_dst, label_dst))
        aug_count += 1

# 合并原始和增强样本
total_sample_info = defaultdict(list)
for cls_name in ID_TO_CLASS.values():
    total_sample_info[cls_name] = sample_info[cls_name] + aug_sample_info[cls_name]
    # 随机打乱
    random.shuffle(total_sample_info[cls_name])
    # 截断到目标数量
    total_sample_info[cls_name] = total_sample_info[cls_name][:CLASS_TARGET[cls_name]]
    print(f"   {cls_name}：最终样本数{len(total_sample_info[cls_name])}")

# -------------------------- 第三步：按目标数量划分样本 --------------------------
# 创建划分后的目录
for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(SPLIT_DATA_DIR, subset, "img"), exist_ok=True)
    os.makedirs(os.path.join(SPLIT_DATA_DIR, subset, "label"), exist_ok=True)

total_target = TARGET_TRAIN_NUM + TARGET_VAL_NUM + TARGET_TEST_NUM
cls_split_num = defaultdict(dict)

# 计算各类别划分数量
for cls_name in ID_TO_CLASS.values():
    total = len(total_sample_info[cls_name])
    train_num = round(total * TARGET_TRAIN_NUM / total_target)
    val_num = round(total * TARGET_VAL_NUM / total_target)
    test_num = total - train_num - val_num
    cls_split_num[cls_name]["train"] = train_num
    cls_split_num[cls_name]["val"] = val_num
    cls_split_num[cls_name]["test"] = test_num

# 全局数量校准
train_diff = TARGET_TRAIN_NUM - sum([v["train"] for v in cls_split_num.values()])
val_diff = TARGET_VAL_NUM - sum([v["val"] for v in cls_split_num.values()])
test_diff = TARGET_TEST_NUM - sum([v["test"] for v in cls_split_num.values()])

sorted_cls = sorted(total_sample_info.keys(), key=lambda x: len(total_sample_info[x]), reverse=True)

# 调整训练集
for i in range(abs(train_diff)):
    cls = sorted_cls[i % len(sorted_cls)]
    if train_diff > 0:
        if cls_split_num[cls]["test"] > 0:
            cls_split_num[cls]["train"] += 1
            cls_split_num[cls]["test"] -= 1
    else:
        if cls_split_num[cls]["train"] > 0:
            cls_split_num[cls]["train"] -= 1
            cls_split_num[cls]["test"] += 1

# 调整验证集
for i in range(abs(val_diff)):
    cls = sorted_cls[i % len(sorted_cls)]
    if val_diff > 0:
        if cls_split_num[cls]["test"] > 0:
            cls_split_num[cls]["val"] += 1
            cls_split_num[cls]["test"] -= 1
    else:
        if cls_split_num[cls]["val"] > 0:
            cls_split_num[cls]["val"] -= 1
            cls_split_num[cls]["test"] += 1

# -------------------------- 第四步：复制文件并生成索引 --------------------------
split_stats = []
subset_index = {"train": [], "val": [], "test": []}
# 记录每个文件对应的类别（用于后续统计）
file_to_cls = {}

print("\n===== 划分数据集 =====")
for cls_name in ID_TO_CLASS.values():
    samples = total_sample_info[cls_name]
    train_num = cls_split_num[cls_name]["train"]
    val_num = cls_split_num[cls_name]["val"]
    test_num = cls_split_num[cls_name]["test"]

    train_samples = samples[:train_num]
    val_samples = samples[train_num:train_num + val_num]
    test_samples = samples[train_num + val_num:train_num + val_num + test_num]

    # 复制训练集
    for img_path, label_path in train_samples:
        img_basename = os.path.basename(img_path)
        img_dst = os.path.join(SPLIT_DATA_DIR, "train", "img", img_basename)
        label_dst = os.path.join(SPLIT_DATA_DIR, "train", "label", os.path.basename(label_path))
        copy2(img_path, img_dst)
        copy2(label_path, label_dst)
        subset_index["train"].append(os.path.join("img", img_basename))
        file_to_cls[img_basename] = cls_name

    # 复制验证集
    for img_path, label_path in val_samples:
        img_basename = os.path.basename(img_path)
        img_dst = os.path.join(SPLIT_DATA_DIR, "val", "img", img_basename)
        label_dst = os.path.join(SPLIT_DATA_DIR, "val", "label", os.path.basename(label_path))
        copy2(img_path, img_dst)
        copy2(label_path, label_dst)
        subset_index["val"].append(os.path.join("img", img_basename))
        file_to_cls[img_basename] = cls_name

    # 复制测试集
    for img_path, label_path in test_samples:
        img_basename = os.path.basename(img_path)
        img_dst = os.path.join(SPLIT_DATA_DIR, "test", "img", img_basename)
        label_dst = os.path.join(SPLIT_DATA_DIR, "test", "label", os.path.basename(label_path))
        copy2(img_path, img_dst)
        copy2(label_path, label_dst)
        subset_index["test"].append(os.path.join("img", img_basename))
        file_to_cls[img_basename] = cls_name

    # 统计
    split_stats.append({
        "Behavior": cls_name,
        "Class ID": [k for k, v in ID_TO_CLASS.items() if v == cls_name][0],
        "Training Set Count": len(train_samples),
        "Validation Set Count": len(val_samples),
        "Test Set Count": len(test_samples),
        "Total Count": len(train_samples) + len(val_samples) + len(test_samples)
    })

# -------------------------- 关键修复：安全截断测试集 --------------------------
print(f"\n===== 调整测试集数量 =====")
print(f"原始测试集数量：{len(subset_index['test'])}")
# 随机打乱测试集（保证随机性）
random.shuffle(subset_index["test"])
# 截断到目标数量
subset_index["test"] = subset_index["test"][:TARGET_TEST_NUM]
print(f"调整后测试集数量：{len(subset_index['test'])}")

# 安全删除多余的测试集文件（只处理文件，跳过文件夹）
test_img_dir = os.path.join(SPLIT_DATA_DIR, "test", "img")
test_label_dir = os.path.join(SPLIT_DATA_DIR, "test", "label")

# 获取保留的图片文件名
keep_img_files = [os.path.basename(path) for path in subset_index["test"]]
keep_label_files = [os.path.splitext(f)[0] + LABEL_SUFFIX for f in keep_img_files]

# 删除多余的图片文件（跳过文件夹）
for file in os.listdir(test_img_dir):
    file_path = os.path.join(test_img_dir, file)
    # 只处理文件，不处理文件夹
    if os.path.isfile(file_path) and file not in keep_img_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"警告：删除图片文件{file}失败：{e}")

# 删除多余的标签文件（跳过文件夹）
for file in os.listdir(test_label_dir):
    file_path = os.path.join(test_label_dir, file)
    # 只处理文件，不处理文件夹
    if os.path.isfile(file_path) and file not in keep_label_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"警告：删除标签文件{file}失败：{e}")

# 重新统计各类别测试集数量
cls_test_count = defaultdict(int)
for path in subset_index["test"]:
    img_basename = os.path.basename(path)
    cls_name = file_to_cls.get(img_basename, "")
    if cls_name:
        cls_test_count[cls_name] += 1

# 更新统计信息
for i, stat in enumerate(split_stats):
    cls_name = stat["Behavior"]
    stat["Test Set Count"] = cls_test_count.get(cls_name, 0)
    stat["Total Count"] = stat["Training Set Count"] + stat["Validation Set Count"] + stat["Test Set Count"]

# -------------------------- 第五步：生成文件 --------------------------
# 生成索引文件
for subset in ["train", "val", "test"]:
    txt_path = os.path.join(SPLIT_DATA_DIR, f"{subset}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(subset_index[subset]))
    print(f"{subset}.txt 已生成")

# 生成统计CSV
stats_df = pd.DataFrame(split_stats)
stats_csv_path = os.path.join(SPLIT_DATA_DIR, "class_count_stats.csv")
stats_df.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
print(f"类别统计CSV已生成")

# -------------------------- 输出结果 --------------------------
print("\n===== 最终结果 ======")
print(f"训练集：{len(subset_index['train'])} 张（目标：{TARGET_TRAIN_NUM}）")
print(f"验证集：{len(subset_index['val'])} 张（目标：{TARGET_VAL_NUM}）")
print(f"测试集：{len(subset_index['test'])} 张（目标：{TARGET_TEST_NUM}）")
print("\n类别详情：")
print(stats_df.to_string(index=False))