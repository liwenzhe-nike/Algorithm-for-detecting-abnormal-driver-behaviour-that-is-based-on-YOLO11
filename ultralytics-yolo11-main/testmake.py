import os
import shutil
import random
from collections import defaultdict

def create_validation_set(image_dir, txt_dir, output_dir, skip_first=1000, target_count_per_class=100, exclude_class="seatbelt"):
    """
    从指定目录的图片中（跳过前 skip_first 张），为每个类别（除了 exclude_class）生成指定数量的验证集。

    :param image_dir: 包含图片的目录
    :param txt_dir: 包含txt标注文件的目录
    :param output_dir: 输出目录，生成的验证集将存放在该目录下
    :param skip_first: 跳过的前N张图片
    :param target_count_per_class: 每个类别的目标数量
    :param exclude_class: 排除的类别名称
    """
    # 创建输出目录
    output_image_dir = os.path.join(output_dir, "images")
    output_txt_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    # 类别映射字典
    class_mapping = {
        "0": "phone",
        "1": "drinking",
        "2": "yawning",
        "3": "seatbelt",
        "4": "drowsy",
        "5": "smoking"
    }

    # 读取所有txt文件并解析分类
    class_files = defaultdict(list)

    # 获取跳过前 skip_first 张图片后的所有图片对应的 txt 文件
    image_files = sorted(os.listdir(image_dir))[skip_first:]
    txt_files = [file.replace(".jpg", ".txt") for file in image_files]

    for txt_file in txt_files:
        if not os.path.exists(os.path.join(txt_dir, txt_file)):
            continue

        txt_path = os.path.join(txt_dir, txt_file)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {txt_path}: {e}")
            continue

        # 检查文件是否为空
        if not lines:
            print(f"Skipping empty file: {txt_file}")
            continue

        # 解析类别索引和边界框信息
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Skipping invalid line in file {txt_file}: {line}")
                continue

            class_index = parts[0]  # 类别索引
            if class_index not in class_mapping:
                print(f"Unknown class index '{class_index}' in file {txt_file}. Skipping...")
                continue

            class_name = class_mapping[class_index]  # 映射到类别名称
            if class_name != exclude_class:
                class_files[class_name].append(txt_file.replace(".txt", ".jpg"))

    # 从每个类别中随机抽取指定数量的文件
    selected_files = []
    for class_name, files in class_files.items():
        if len(files) > target_count_per_class:
            selected_files.extend(random.sample(files, target_count_per_class))
        else:
            selected_files.extend(files)  # 如果类别数量不足，直接全部选取

    # 去重
    selected_files = list(set(selected_files))

    # 复制选中的文件到输出目录
    for file in selected_files:
        # 复制图片文件
        shutil.copy(os.path.join(image_dir, file), os.path.join(output_image_dir, file))
        # 复制对应的txt文件
        txt_file = file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(txt_dir, txt_file), os.path.join(output_txt_dir, txt_file))

    print(f"Validation set generated successfully!")
    print(f"Each class has up to {target_count_per_class} samples.")
    print(f"Total samples: {len(selected_files)}")
    print(f"Images saved to: {output_image_dir}")
    print(f"Labels saved to: {output_txt_dir}")


if __name__ == "__main__":
    # 示例用法
    image_directory = "D:/地平线4/yolotest/train/images"  # 替换为你的图片目录
    txt_directory = "D:/地平线4/yolotest/train/labels"  # 替换为你的txt标注目录
    output_directory = "D:/地平线4/yolo11/test/"  # 替换为输出目录
    skip_first = 1000  # 跳过的前N张图片
    target_count_per_class = 100  # 每个类别的目标数量
    exclude_class = "seatbelt"  # 排除的类别名称

    create_validation_set(image_directory, txt_directory, output_directory, skip_first, target_count_per_class, exclude_class)