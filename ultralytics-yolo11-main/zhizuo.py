from ultralytics import YOLO
import torch

# 加载各消融变体模型
models = {
    "A": YOLO("roscsa.yaml"),

}

# 计算并打印每个变体的参数量
for name, model in models.items():
    # 统计可训练参数总数
    total_params = sum(p.numel() for p in model.model.parameters())
    # 转换为百万（M），保留2位小数
    total_params_m = round(total_params / 1e6, 2)
    print(f"Variant {name}: Parameter counts = {total_params_m}M")

    # 可选：统计可训练参数（排除冻结层）
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    trainable_params_m = round(trainable_params / 1e6, 2)
    print(f"Variant {name}: Trainable parameters = {trainable_params_m}M")