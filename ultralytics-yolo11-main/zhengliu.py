import yaml
from graphviz import Digraph

def parse_yaml(file_path):
    """解析 YOLO 的 .yaml 文件，提取模型结构信息"""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def draw_model_structure(config, output_file="model_structure.png"):
    """根据 YOLO 的配置信息绘制模型结构图"""
    dot = Digraph(comment="YOLO Model Structure")

    # 遍历模型的每个模块
    for idx, module in enumerate(config['backbone'] + config['head']):
        module_type = module['type']
        module_name = f"{module_type}_{idx}"
        dot.node(module_name, f"{module_type}\n{module.get('args', '')}")

        # 如果不是第一个模块，添加边
        if idx > 0:
            prev_module_name = f"{prev_module_type}_{idx - 1}"
            dot.edge(prev_module_name, module_name)

        prev_module_type = module_type

    # 保存并渲染图形
    dot.render(output_file, view=True)

# 示例：解析 YOLO 的 .yaml 文件并绘制结构图
yaml_file = "D:/BaiduNetdiskDownload/yolo11train/yolo11魔鬼面具最新版/ultralytics-yolo11-main/ultralytics/cfg/models/11/roscsa.yaml"  # 替换为你的 YOLO .yaml 文件路径
config = parse_yaml(yaml_file)
draw_model_structure(config)