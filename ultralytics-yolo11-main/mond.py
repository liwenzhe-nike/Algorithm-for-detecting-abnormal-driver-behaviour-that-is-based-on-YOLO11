from ultralytics import YOLO
import torch
import os
import torch.nn as nn
from collections import defaultdict
from ultralytics.nn.modules import Conv, Bottleneck, Concat
from ultralytics.nn.extra_modules.block import EMSConvP, C3k2_EMBC, C3k2_EMSCP


class ChannelAwarePruner:
    def __init__(self, model_path, prune_factor=0.5):
        self.model = YOLO(model_path).model
        self.prune_factor = prune_factor
        self.channel_map = {}
        self.dependency_graph = defaultdict(list)
        self.concat_sources = {}
        self.protected_layers = {
            'model.0.conv': 3,  # 强制保留输入通道
            'model.2': 64,
            'model.4': 128,
            'model.7': 256,
            'model.14': 512
        }
        self.min_channels = 16  # 全局最小通道数

        self._parse_model_config()
        self._build_full_dependency()
        self._init_channel_records()

    def _parse_model_config(self):
        """增强型模型结构解析"""
        try:
            for i, layer in enumerate(self.model.model):
                if isinstance(layer, Concat):
                    sources = getattr(layer, 'from_idx', [])
                    if not sources and i < len(self.model.yaml):
                        args = self.model.yaml[i][1]
                        sources = args[0] if isinstance(args, list) else []
                    self.concat_sources[layer] = [i + x if x < 0 else x for x in sources]
        except Exception as e:
            print(f"⚠️ 配置解析异常: {str(e)}")

    def _build_full_dependency(self):
        """三维依赖关系构建"""
        parent_map = defaultdict(list)
        for name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                parent_map[child].append(module)
                self.dependency_graph[module].append(child)

        # 构建逆向依赖
        for concat_layer, sources in self.concat_sources.items():
            for src_idx in sources:
                if src_idx < len(self.model.model):
                    src_module = self.model.model[src_idx]
                    self.dependency_graph[src_module].append(concat_layer)

    def _init_channel_records(self):
        """通道记录系统初始化"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, Concat)):
                self.channel_map[name] = self._get_valid_channels(module)

    def _get_valid_channels(self, module):
        """安全获取通道数"""
        if isinstance(module, nn.Conv2d):
            return module.out_channels
        if isinstance(module, Concat):
            return sum(self.channel_map.get(name, 0) for name in self._get_concat_source_names(module))
        return 0

    def _get_concat_source_names(self, concat_layer):
        """获取Concat层输入源名称"""
        sources = []
        for idx in self.concat_sources.get(concat_layer, []):
            if idx < len(self.model.model):
                src_module = self.model.model[idx]
                sources.append(self._get_layer_name(src_module))
        return sources

    def auto_prune(self, save_path):
        """全自动剪枝流程"""
        try:
            self._calculate_threshold()
            self._prune_all_modules()
            self._propagate_all_changes()
            self._final_consistency_check()
            self._save_model(save_path)
        except Exception as e:
            print(f"❌ 剪枝失败: {str(e)}")
            self._generate_diagnosis()

    def _calculate_threshold(self):
        """动态阈值计算"""
        bn_weights = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_weights.append(m.weight.abs().detach())
        sorted_weights = torch.sort(torch.cat(bn_weights), descending=True)[0]
        self.threshold = sorted_weights[int(len(sorted_weights) * self.prune_factor)]

    def _prune_all_modules(self):
        """全局剪枝入口"""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, (Bottleneck, C3k2_EMBC, C3k2_EMSCP)):
                self._prune_complex_module(module, name)
            elif isinstance(module, nn.Conv2d):
                self._prune_single_conv(module, name)

    def _prune_complex_module(self, module, name):
        """智能处理复杂模块"""
        if 'Bottleneck' in str(type(module)):
            print(f"🔧 处理瓶颈模块: {name}")
            for child_name, child in module.named_children():
                if 'act' not in child_name and 'bn' not in child_name:
                    self._prune_single_conv(child, f"{name}.{child_name}")
            return

        print(f"🛠️ 处理复杂模块: {name}")
        for child_name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, Bottleneck)):
                self._prune_single_conv(child, f"{name}.{child_name}")
            else:
                self._prune_complex_module(child, f"{name}.{child_name}")

    def _prune_single_conv(self, conv, name):
        """安全剪枝逻辑"""
        if name in self.protected_layers:
            new_channels = self.protected_layers[name]
            print(f"🛡️ 保护层重置通道 [{name}] → {new_channels}")
            conv.out_channels = new_channels
            self.channel_map[name] = new_channels
            return

        if not hasattr(conv, 'weight'):
            print(f"⏩ 跳过无权重模块: {name}")
            return

        # 深度搜索BN层（支持3层嵌套）
        bn_layers = []

        def _deep_find_bn(m, depth=0):
            if depth > 3: return
            if isinstance(m, nn.BatchNorm2d): bn_layers.append(m)
            for child in m.children(): _deep_find_bn(child, depth + 1)

        _deep_find_bn(conv)

        if bn_layers:
            self._prune_with_bn(conv, name, bn_layers)
        else:
            self._prune_by_weight(conv, name)

    def _prune_with_bn(self, conv, name, bn_layers):
        """标准BN剪枝"""
        main_bn = bn_layers[0]
        bn_weights = main_bn.weight.abs().detach()

        min_channels = max(self.min_channels, self.protected_layers.get(name, 0))
        keep_idx = torch.where(bn_weights >= self.threshold)[0]

        if len(keep_idx) < min_channels:
            keep_idx = torch.argsort(bn_weights, descending=True)[:min_channels]
        keep_idx = keep_idx[keep_idx < bn_weights.size(0)]

        # 执行剪枝
        conv.out_channels = len(keep_idx)
        conv.weight = nn.Parameter(conv.weight[keep_idx])
        for bn in bn_layers:
            self._update_bn_params(bn, keep_idx)

        self.channel_map[name] = len(keep_idx)
        print(f"✂️ 剪枝完成 [{name}] 保留通道: {len(keep_idx)}")

    def _prune_by_weight(self, conv, name):
        """权重剪枝备用方案"""
        weight_importance = torch.mean(conv.weight.abs(), dim=(1, 2, 3))
        sorted_idx = torch.argsort(weight_importance, descending=True)

        min_channels = max(self.min_channels, self.protected_layers.get(name, 0))
        keep_idx = sorted_idx[:max(int(len(sorted_idx) * 0.5), min_channels)]

        conv.out_channels = len(keep_idx)
        conv.weight = nn.Parameter(conv.weight[keep_idx])
        self.channel_map[name] = len(keep_idx)
        print(f"⚖️ 权重剪枝完成 [{name}] 保留通道: {len(keep_idx)}")

    def _update_bn_params(self, bn, indices):
        """安全更新BN参数"""
        valid_indices = indices[indices < bn.num_features]
        if len(valid_indices) == 0:
            valid_indices = torch.arange(bn.num_features)[:self.min_channels]

        bn.num_features = len(valid_indices)
        bn.weight = nn.Parameter(bn.weight.data[valid_indices])
        bn.bias = nn.Parameter(bn.bias.data[valid_indices])
        bn.running_mean = bn.running_mean[valid_indices]
        bn.running_var = bn.running_var[valid_indices]

    def _propagate_all_changes(self):
        """全局通道传播"""
        print("\n🌐 开始通道传播")
        visited = set()
        for name in list(self.channel_map.keys()):
            module = self._get_module_by_name(name)
            if module and module not in visited:
                self._propagate_changes(module, visited)

    def _propagate_changes(self, module, visited):
        """递归传播变更"""
        if module in visited: return
        visited.add(module)

        if isinstance(module, nn.Conv2d):
            self._update_dependent_layers(module)

        for dependent in self.dependency_graph.get(module, []):
            self._propagate_changes(dependent, visited)

    def _update_dependent_layers(self, conv):
        """更新依赖该卷积的层"""
        current_out = max(self.channel_map.get(self._get_layer_name(conv), conv.out_channels), self.min_channels)
        print(f"🔧 传播通道 [{self._get_layer_name(conv)}] → {current_out}")

        for dependent in self.dependency_graph.get(conv, []):
            if isinstance(dependent, nn.Conv2d):
                self._update_conv_input(dependent, current_out)
            elif isinstance(dependent, Concat):
                self._update_concat_layer(dependent)

    def _update_conv_input(self, conv, target_in):
        """安全更新输入通道"""
        conv_name = self._get_layer_name(conv)
        target_in = max(target_in, self.min_channels)

        if conv.in_channels == target_in:
            return

        print(f"🔄 更新输入通道 [{conv_name}] {conv.in_channels}→{target_in}")

        # 处理分组卷积
        groups = conv.groups
        if groups > 1:
            target_in = target_in * groups

        # 智能权重迁移
        new_weight = torch.zeros(conv.out_channels, target_in, *conv.weight.shape[2:],
                                 device=conv.weight.device)
        min_c = min(target_in, conv.weight.shape[1])
        new_weight[:, :min_c] = conv.weight[:, :min_c]

        conv.weight = nn.Parameter(new_weight)
        conv.in_channels = target_in
        self.channel_map[conv_name] = conv.out_channels

    def _update_concat_layer(self, concat_layer):
        """动态更新Concat层"""
        concat_name = self._get_layer_name(concat_layer)
        total_channels = sum(
            self.channel_map.get(src_name, 0)
            for src_name in self._get_concat_source_names(concat_layer)
        )
        total_channels = max(total_channels, self.min_channels)

        concat_layer.out_channels = total_channels
        self.channel_map[concat_name] = total_channels
        print(f"🔗 更新Concat层 [{concat_name}] 总通道: {total_channels}")

    def _final_consistency_check(self):
        """最终一致性验证"""
        print("\n🔍 执行最终验证")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 输入通道验证
                expected_in = self._get_expected_input(name)
                expected_in = max(expected_in, self.min_channels)

                if module.in_channels != expected_in:
                    print(f"⚡ 修复输入通道 [{name}] {module.in_channels}→{expected_in}")
                    module.in_channels = expected_in

                # 输出通道验证
                expected_out = max(self.channel_map.get(name, module.out_channels), self.min_channels)
                if module.out_channels != expected_out:
                    print(f"⚡ 修复输出通道 [{name}] {module.out_channels}→{expected_out}")
                    module.out_channels = expected_out

    def _get_expected_input(self, name):
        """智能获取预期输入通道"""
        # 查找所有可能的上游层
        for parent, children in self.dependency_graph.items():
            parent_name = self._get_layer_name(parent)
            if any(self._get_layer_name(child) == name for child in children):
                return self.channel_map.get(parent_name, 0)
        return 0

    def _get_layer_name(self, module):
        """获取模块的完整名称"""
        for name, m in self.model.named_modules():
            if m is module:
                return name
        return "unknown"

    def _save_model(self, save_path):
        """模型保存与验证"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

        try:
            test_input = torch.randn(1, 3, 640, 640)
            self.model(test_input)
            print(f"✅ 验证成功！模型保存至: {save_path}")
        except Exception as e:
            print(f"❌ 验证失败: {str(e)}")
            self._generate_diagnosis()

    def _generate_diagnosis(self):
        """生成深度诊断报告"""
        print("\n🔍 深度诊断报告:")
        # 检查所有卷积层
        conv_issues = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                try:
                    actual_in = module.weight.shape[1] * module.groups
                    if actual_in != module.in_channels:
                        conv_issues.append(f"{name.ljust(30)} 输入: {module.in_channels}≠{actual_in}")
                except:
                    pass

        print(f"卷积层异常 ({len(conv_issues)} 处):")
        for issue in conv_issues[:5]:
            print(issue)


if __name__ == "__main__":
    pruner = ChannelAwarePruner(
        model_path=r'D:/BaiduNetdiskDownload/yolo11train/yolo11魔鬼面具最新版/ultralytics-yolo11-main/runs/train/exp111/weights/best.pt',
        prune_factor=0.5
    )
    pruner.auto_prune(r'pruned_model.pt')