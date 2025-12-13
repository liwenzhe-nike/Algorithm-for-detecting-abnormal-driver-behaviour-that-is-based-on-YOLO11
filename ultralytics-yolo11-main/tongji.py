import os

# 指定目录路径
directory_path = 'D:/BaiduNetdiskDownload/右上角度很大数据集/label'

# 初始化类别计数字典
class_counts = {}

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)  # 获取文件的完整路径
    if os.path.isfile(file_path):  # 确保是文件
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split()
                    if parts:
                        class_name = parts[0]
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")

# 输出结果
print("类别总数:", len(class_counts))
print("每个类别的样本数量:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")