import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO  # 假设你使用的是 YOLOv11 的相关库

# 加载模型
model_path = "D:/BaiduNetdiskDownload/yolo11train/yolo11魔鬼面具最新版/ultralytics-yolo11-main/runs/train/exp111/weights/best.pt"
model = YOLO(model_path)

# 打印模型结构
print("模型结构:")
print(model)

# 检查模型的每一层输入输出形状
def check_model_layers(model, input_tensor):
    print("\n检查模型的每一层输入输出形状:")
    x = input_tensor
    print("输入张量形状:", x.shape)
    for i, layer in enumerate(model.model.children()):
        try:
            x = layer(x)
            print(f"经过层 {i} ({layer.__class__.__name__}): 输入形状 {x.shape}")
        except Exception as e:
            print(f"在层 {i} ({layer.__class__.__name__}) 出现错误: {e}")
            break

# 创建一个虚拟输入张量
# 假设输入是 3 通道的 RGB 图像，大小为 640x640
input_tensor = torch.randn(1, 3, 640, 640)

# 检查模型的每一层输入输出形状
check_model_layers(model, input_tensor)

# 检查输入数据的预处理
def check_input_data(data_path):
    print("\n检查输入数据的预处理:")
    # 假设输入数据是图像文件
    image = Image.open(data_path)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 调整图像大小
        transforms.ToTensor(),         # 转换为张量
    ])
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
    print("输入数据形状:", input_tensor.shape)
    return input_tensor

# 检查实际输入数据的形状
data_path = "path_to_your_image.jpg"  # 替换为你的图像路径
input_tensor = check_input_data(data_path)

# 使用实际输入数据运行模型
print("\n使用实际输入数据运行模型:")
try:
    output = model(input_tensor)
    print("模型输出形状:", output.shape)
except Exception as e:
    print(f"运行模型时出现错误: {e}")