import os
import random
import pandas as pd
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import uuid

# -------------------------- 核心配置（与之前一致） --------------------------
SUBSET_NUM = {
    "train": 2099,
    "test": 300,
    "val": 598
}
CLASS_RATIO = {
    0: 1.0,  # Phone
    1: 1.0,  # drinking
    2: 1.0,  # yawning
    3: 1.333,  # seatblt
    4: 1.0,  # drowsy
    5: 1.0  # smoking
}
ID_TO_CLASS = {0: "Phone", 1: "drinking", 2: "yawning", 3: "seatblt", 4: "drowsy", 5: "smoking"}
CLASS_TO_ID = {v: k for k, v in ID_TO_CLASS.items()}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------- 工具函数 --------------------------
def read_label(label_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            with open(label_path, 'r', encoding=enc) as f:
                lines = [line.strip() for line in f if line.strip() and len(line.strip().split()) >= 5]
            class_ids = [int(line.split()[0]) for line in lines if int(line.split()[0]) in ID_TO_CLASS]
            return class_ids, lines
        except Exception:
            continue
    return [], []


def collect_all_samples(original_dir):
    all_samples = []
    class_sample_map = defaultdict(list)

    print("===== 收集所有样本 =====")
    print(f"扫描目录：{original_dir}")

    for root, dirs, files in os.walk(original_dir):
        files.sort()
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(root, file)
                label_paths = [
                    os.path.join(root, os.path.splitext(file)[0] + ".txt"),
                    os.path.join(root.replace("img", "label"), os.path.splitext(file)[0] + ".txt"),
                    os.path.join(root.replace("image", "label"), os.path.splitext(file)[0] + ".txt"),
                    os.path.join(root.replace("images", "labels"), os.path.splitext(file)[0] + ".txt"),
                    os.path.join(os.path.dirname(root), "labels", os.path.splitext(file)[0] + ".txt")
                ]
                label_path = None
                for lp in label_paths:
                    if os.path.exists(lp):
                        label_path = lp
                        break
                if not label_path:
                    continue

                class_ids, label_lines = read_label(label_path)
                if not class_ids:
                    continue

                sample = (img_path, label_path, class_ids, label_lines)
                all_samples.append(sample)
                for cid in class_ids:
                    class_sample_map[cid].append(sample)

    print(f"总样本数：{len(all_samples)}")
    for cid in ID_TO_CLASS:
        print(f"{ID_TO_CLASS[cid]}：{len(class_sample_map[cid])}个样本")

    if len(all_samples) == 0:
        print("错误：未收集到任何样本，请检查数据集路径！")
    return all_samples, class_sample_map


def split_samples_by_order(all_samples):
    train_num = SUBSET_NUM["train"]
    val_num = SUBSET_NUM["val"]
    test_num = SUBSET_NUM["test"]

    print(f"\n按顺序划分：")
    train_samples = all_samples[:train_num]
    val_samples = all_samples[train_num:train_num + val_num]
    test_samples = all_samples[train_num + val_num:train_num + val_num + test_num]

    if len(train_samples) < train_num:
        train_samples.extend(random.choices(train_samples, k=train_num - len(train_samples)))
    if len(val_samples) < val_num:
        val_samples.extend(random.choices(val_samples, k=val_num - len(val_samples)))
    if len(test_samples) < test_num:
        test_samples.extend(random.choices(test_samples, k=test_num - len(test_samples)))

    print(f"train：{len(train_samples)}个样本")
    print(f"val：{len(val_samples)}个样本")
    print(f"test：{len(test_samples)}个样本")
    return {"train": train_samples, "val": val_samples, "test": test_samples}


def augment_image_opencv(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.randint(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    brightness = random.uniform(-0.2, 0.2)
    img = cv2.convertScaleAbs(img, alpha=1, beta=brightness * 255)
    return img


def get_unique_filename(base_path, suffix="jpg"):
    unique_id = str(uuid.uuid4())[:8]
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    return f"{base_name}_{unique_id}.{suffix}"


def augment_sample(sample, save_dir, subset):
    img_path, label_path, class_ids, label_lines = sample
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_dir = os.path.join(save_dir, subset, "images")
    label_dir = os.path.join(save_dir, subset, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    aug_img = augment_image_opencv(img)
    aug_img_name = get_unique_filename(img_path, "jpg")
    aug_label_name = get_unique_filename(label_path, "txt")

    aug_img_path = os.path.join(img_dir, aug_img_name)
    aug_label_path = os.path.join(label_dir, aug_label_name)

    cv2.imwrite(aug_img_path, aug_img)
    with open(aug_label_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(label_lines))
    return (aug_img_path, aug_label_path, class_ids, label_lines)


def calculate_ideal_count():
    total_weight = sum(CLASS_RATIO.values())
    total_samples = sum(SUBSET_NUM.values())
    subset_weight = {
        "train": SUBSET_NUM["train"] / total_samples,
        "test": SUBSET_NUM["test"] / total_samples,
        "val": SUBSET_NUM["val"] / total_samples
    }

    ideal_count = defaultdict(dict)
    for cid in CLASS_RATIO:
        for subset in subset_weight:
            ideal = int(SUBSET_NUM[subset] * (CLASS_RATIO[cid] / total_weight))
            ideal_count[cid][subset] = ideal

    for subset in subset_weight:
        total_ideal = sum(ideal_count[cid][subset] for cid in ideal_count)
        diff = SUBSET_NUM[subset] - total_ideal
        if diff > 0:
            for cid in ideal_count:
                add = int(diff * (CLASS_RATIO[cid] / total_weight))
                ideal_count[cid][subset] += add
                diff -= add
                if diff == 0:
                    break
            if diff > 0:
                ideal_count[random.choice(list(ideal_count.keys()))][subset] += diff
    return ideal_count


def balance_class_in_subset(subset_samples, subset_name, ideal_count, save_dir):
    class_sample_map = defaultdict(list)
    for sample in subset_samples:
        for cid in sample[2]:
            class_sample_map[cid].append(sample)

    balanced_samples = []
    class_count = defaultdict(int)

    for cid in ideal_count:
        target = ideal_count[cid][subset_name]
        if target == 0:
            continue
        samples = class_sample_map.get(cid, [])
        if not samples:
            print(f"警告：{ID_TO_CLASS[cid]}在{subset_name}中无样本，跳过")
            continue

        current = []
        while len(current) < target:
            take = min(len(samples), target - len(current))
            current.extend(random.sample(samples, take))
            if len(current) < target:
                sample = random.choice(samples)
                aug = augment_sample(sample, save_dir, subset_name)
                if aug:
                    current.append(aug)

        current = current[:target]
        balanced_samples.extend(current)
        class_count[cid] = len(current)

    if len(balanced_samples) > SUBSET_NUM[subset_name]:
        balanced_samples = random.sample(balanced_samples, SUBSET_NUM[subset_name])
    elif len(balanced_samples) < SUBSET_NUM[subset_name]:
        need = SUBSET_NUM[subset_name] - len(balanced_samples)
        balanced_samples.extend(random.choices(balanced_samples, k=need))

    return balanced_samples, class_count


def save_samples(subset_samples, save_dir):
    for subset in subset_samples:
        img_dir = os.path.join(save_dir, subset, "images")
        label_dir = os.path.join(save_dir, subset, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        index_lines = []
        saved_files = set()
        for sample in subset_samples[subset]:
            img_path, label_path, class_ids, label_lines = sample
            img_name = get_unique_filename(img_path, "jpg")
            label_name = get_unique_filename(label_path, "txt")

            target_img_path = os.path.join(img_dir, img_name)
            target_label_path = os.path.join(label_dir, label_name)

            if img_name not in saved_files:
                shutil.copy(img_path, target_img_path)
                shutil.copy(label_path, target_label_path)
                saved_files.add(img_name)
                index_lines.append(os.path.abspath(target_img_path))

        if index_lines:
            with open(os.path.join(save_dir, f"{subset}.txt"), 'w', encoding='utf-8') as f:
                f.write("\n".join(index_lines))


def plot_class_distribution(stats_df):
    """生成你提供的类似柱状图"""
    behaviors = stats_df["Behavior"]
    train_counts = stats_df["Training Set Count"]
    val_counts = stats_df["Validation Set Count"]
    test_counts = stats_df["Test Set Count"]

    x = np.arange(len(behaviors))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, train_counts, width, label='Training')
    rects2 = ax.bar(x, val_counts, width, label='Validation')
    rects3 = ax.bar(x + width, test_counts, width, label='Test')

    ax.set_ylabel('Instances')
    ax.set_title('Class Distribution Across Subsets')
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors)
    ax.legend()

    fig.tight_layout()
    plt.show()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    ORIGINAL_DATA_DIR = r"E:/BaiduNetdiskDownload/YOLO"  # 替换为你的数据集路径
    FINAL_DATA_DIR = r".data2228"  # 输出目录

    try:
        all_samples, class_sample_map = collect_all_samples(ORIGINAL_DATA_DIR)
        if len(all_samples) == 0:
            exit(1)

        ordered_subsets = split_samples_by_order(all_samples)
        ideal_count = calculate_ideal_count()

        balanced_subsets = {}
        subset_count = defaultdict(dict)
        for subset in ordered_subsets:
            print(f"\n平衡{subset}子集...")
            balanced, count = balance_class_in_subset(ordered_subsets[subset], subset, ideal_count, FINAL_DATA_DIR)
            balanced_subsets[subset] = balanced
            subset_count[subset] = count

        save_samples(balanced_subsets, FINAL_DATA_DIR)

        # 生成你喜欢的统计表格
        stats = []
        for cls in ID_TO_CLASS.values():
            cid = CLASS_TO_ID[cls]
            train = subset_count["train"].get(cid, 0)
            val = subset_count["val"].get(cid, 0)
            test = subset_count["test"].get(cid, 0)
            stats.append({
                "Behavior": cls,
                "Training Set Count": train,
                "Validation Set Count": val,
                "Test Set Count": test,
                "Total Count": train + val + test
            })
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(FINAL_DATA_DIR, "stats.csv"), index=False, encoding='utf-8-sig')

        # 输出结果
        print("\n" + "=" * 80)
        print("你喜欢的数据集统计结果：")
        print("=" * 80)
        print(stats_df.to_string(index=False))
        print("=" * 80)

        # 生成类似的柱状图
        plot_class_distribution(stats_df)

    except Exception as e:
        print(f"错误：{e}")
        import traceback

        traceback.print_exc()