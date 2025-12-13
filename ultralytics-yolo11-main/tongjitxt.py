import os
from collections import defaultdict

def count_classes(txt_dir):
    """
    统计txt标注文件中每个类别的数量。

    :param txt_dir: 包含txt标注文件的目录
    """
    # 类别映射字典
    class_mapping = {
        "0": "phone",
        "1": "drinking",
        "2": "yawning",
        "3": "seatbelt",
        "4": "drowsy",
        "5": "smoking"
    }

    # 用于统计每个类别的出现次数
    class_counts = defaultdict(int)

    for txt_file in os.listdir(txt_dir):
        if not txt_file.endswith(".txt"):
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
            class_counts[class_name] += 1  # 统计类别出现次数

    # 打印每个类别的出现次数
    print("\nClass counts in the dataset:")
    for class_name, count in class_counts.items():
        print(f"Class '{class_name}': {count} instances")


if __name__ == "__main__":
    # 示例用法
    txt_directory = "D:/地平线4/yolo11/train/labels"  # 替换为你的txt标注目录
    count_classes(txt_directory)