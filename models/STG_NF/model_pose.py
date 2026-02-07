"""
STG-NF model, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch
"""

import math
import torch
import torch.nn as nn

from models.STG_NF.modules_pose import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from models.STG_NF.utils import split_feature
from models.STG_NF.graph import Graph
from models.STG_NF.stgcn import st_gcn


def nan_throw(tensor, name="tensor"):
    stop = False
    if ((tensor != tensor).any()):
        print(name + " has nans")
        stop = True
    if (torch.isinf(tensor).any()):
        print(name + " has infs")
        stop = True
    if stop:
        print(name + ": " + str(tensor))


def get_stgcn(in_channels, hidden_channels, out_channels,
              temporal_kernel_size=9, spatial_kernel_size=2, first=False):
    kernel_size = (temporal_kernel_size, spatial_kernel_size)
    if hidden_channels == 0:
        block = nn.ModuleList((
            st_gcn(in_channels, out_channels, kernel_size, 1, residual=(not first)),
        ))
    else:
        block = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=(not first)),
            st_gcn(hidden_channels, out_channels, kernel_size, 1, residual=(not first)),
        ))

    return block


def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class Adapter(nn.Module):
    """
    用于增量学习的Adapter模块
    直接复用已有的FlowStep类，确保与模型核心组件的一致性
    采用残差连接方式，保持轻量级和可插拔性
    """
    def __init__(
            self,
            in_channels,
            hidden_dim=0,
            actnorm_scale=1.0,
            flow_permutation="invconv",
            flow_coupling="additive",
            LU_decomposed=True,
            temporal_kernel_size=4,
            edge_importance_weighting=False,
            strategy='uniform',
            max_hops=8,
            device='cuda:0'
    ):
        super().__init__()
        
        self.device = device
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # 直接复用FlowStep类作为adapter的核心组件
        # 使用较小的hidden_dim以保持轻量级
        self.flow_step = FlowStep(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            temporal_kernel_size=temporal_kernel_size,
            edge_importance_weighting=edge_importance_weighting,
            last=False,
            first=True,
            strategy=strategy,
            max_hops=max_hops,
            device=device
        )
        
        # 初始化FlowStep参数，确保adapter初始状态接近恒等变换
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化adapter参数，使初始状态接近恒等变换"""
        # 对FlowStep中的耦合层参数进行初始化，使其接近恒等变换
        for name, param in self.flow_step.named_parameters():
            if 'block' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=0.01)  # 使用小gain确保初始输出接近0
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x):
        """前向传播，采用残差连接方式使用FlowStep"""
        # 保存输入用于残差连接
        residual = x
        
        # 使用FlowStep的normal_flow进行变换
        # adapter不贡献logdet，所以忽略返回的logdet
        flow_output, _ = self.flow_step.normal_flow(x, logdet=0.0)
        
        # 残差连接：将FlowStep的输出与原始输入相加
        return residual + flow_output


class AdapterPool:
    """
    Adapter池，用于管理多个adapter实例
    支持基于均值相似度的查询和动态更新
    """
    def __init__(
            self,
            max_size=10,
            similarity_threshold=0.1,
            in_channels=0,
            adapter_hidden_dim=64,
            adapter_dropout=0.1,
            device='cuda:0'
    ):
        """
        参数:
            max_size: adapter池的最大容量
            similarity_threshold: 相似度阈值，低于此阈值的adapter被认为是不同的
            in_channels: 输入通道数（必须与FlowStep输出通道数相同）
            adapter_hidden_dim: adapter隐藏层维度
            adapter_dropout: adapter的dropout率
            device: 设备
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.in_channels = in_channels
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_dropout = adapter_dropout
        self.device = device
        
        # adapter池，格式为：[(mean_key, adapter_instance), ...]
        self.pool = []
        
    def add_adapter(self, adapter=None, target_channels=None, flowstep_params=None):
        """
        向池中添加adapter
        如果adapter为None，则创建一个新的adapter
        参数:
            adapter: 要添加的adapter实例，如果为None则创建新实例
            target_channels: 目标通道数（必须与FlowStep输出通道数相同）
            flowstep_params: FlowStep参数，用于创建与FlowStep结构一致的adapter
        """
        if adapter is None:
            # 使用指定的目标通道数或默认的in_channels
            channels = target_channels if target_channels is not None else self.in_channels
            
            # 如果提供了FlowStep参数，则创建与FlowStep结构一致的adapter
            if flowstep_params is not None:
                adapter = Adapter(
                    in_channels=channels,
                    hidden_dim=self.adapter_hidden_dim,
                    actnorm_scale=flowstep_params.get('actnorm_scale', 1.0),
                    flow_permutation=flowstep_params.get('flow_permutation', "invconv"),
                    flow_coupling=flowstep_params.get('flow_coupling', "additive"),
                    LU_decomposed=flowstep_params.get('LU_decomposed', True),
                    temporal_kernel_size=flowstep_params.get('temporal_kernel_size', 4),
                    edge_importance_weighting=flowstep_params.get('edge_importance_weighting', False),
                    strategy=flowstep_params.get('strategy', 'uniform'),
                    max_hops=flowstep_params.get('max_hops', 8),
                    device=self.device
                ).to(self.device)
            else:
                # 使用默认参数创建adapter
                adapter = Adapter(
                    in_channels=channels,
                    hidden_dim=self.adapter_hidden_dim,
                    device=self.device
                ).to(self.device)
        
        # 计算adapter的均值作为key
        mean_key = self._get_adapter_key(adapter)
        
        # 检查池中是否已存在相似的adapter
        for i, (existing_key, existing_adapter) in enumerate(self.pool):
            if abs(mean_key - existing_key) < self.similarity_threshold:
                # 已存在相似的adapter，替换它
                self.pool[i] = (mean_key, adapter)
                return i
        
        # 池中不存在相似的adapter，添加新的
        if len(self.pool) >= self.max_size:
            # 池已满，移除最旧的adapter
            self.pool.pop(0)
        
        self.pool.append((mean_key, adapter))
        return len(self.pool) - 1
    
    def _get_adapter_key(self, adapter):
        """
        计算adapter的均值作为key
        """
        with torch.no_grad():
            # 收集所有参数的均值
            param_means = []
            for param in adapter.parameters():
                param_means.append(param.mean().item())
            # 计算所有参数均值的平均值作为adapter的key
            return sum(param_means) / len(param_means)
    
    def query_adapter(self, query_key, return_index=False):
        """
        根据查询key查找最相似的adapter
        参数:
            query_key: 查询的key（均值）
            return_index: 是否返回adapter在池中的索引
        返回:
            最相似的adapter实例，如果return_index为True，则返回(adapter, index)
        """
        if not self.pool:
            return None
        
        # 计算所有adapter与查询key的相似度
        similarities = [abs(mean_key - query_key) for mean_key, _ in self.pool]
        
        # 找到最相似的adapter
        min_index = similarities.index(min(similarities))
        best_adapter = self.pool[min_index][1]
        
        if return_index:
            return best_adapter, min_index
        else:
            return best_adapter
    
    def get_adapter_by_index(self, index):
        """
        根据索引获取adapter
        """
        if index < 0 or index >= len(self.pool):
            return None
        return self.pool[index][1]
    
    def get_all_adapters(self):
        """
        获取池中所有的adapter
        """
        return [adapter for _, adapter in self.pool]
    
    def clear(self):
        """
        清空adapter池
        """
        self.pool = []
    
    def size(self):
        """
        获取池中adapter的数量
        """
        return len(self.pool)


class FlowStep(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            A=None,
            temporal_kernel_size=4,
            edge_importance_weighting=False,
            last=False,
            first=False,
            strategy='uniform',
            max_hops=8,
            device='cuda:0',
            use_adapter=False,
            adapter_hidden_dim=0
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.flow_coupling = flow_coupling
        self.use_adapter = use_adapter
        
        if A is None:
            g = Graph(strategy=strategy, max_hop=max_hops)
            self.A = torch.from_numpy(g.A).float().to(device)

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_stgcn(in_channels // 2, in_channels // 2, hidden_channels,
                                   temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=self.A.size(0),
                                   first=first
                                   )
        elif flow_coupling == "affine":
            self.block = get_stgcn(in_channels // 2, hidden_channels, in_channels,
                                   temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=self.A.size(0),
                                   first=first)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.block
            ])
        else:
            self.edge_importance = [1] * len(self.block)
        
        # 4. 添加adapter（如果启用）
        if self.use_adapter:
            self.adapter = Adapter(
                in_channels=in_channels,
                hidden_dim=adapter_hidden_dim,
                actnorm_scale=actnorm_scale,
                flow_permutation=flow_permutation,
                flow_coupling=flow_coupling,
                LU_decomposed=LU_decomposed,
                temporal_kernel_size=temporal_kernel_size,
                edge_importance_weighting=edge_importance_weighting,
                strategy=strategy,
                max_hops=max_hops,
                device=device
            )

    def forward(self, input, logdet=None, reverse=False, label=None):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            if len(z1.shape) == 3:
                z1 = z1.unsqueeze(dim=1)
            if len(z2.shape) == 3:
                z2 = z2.unsqueeze(dim=1)
            h = z1.clone()
            for gcn, importance in zip(self.block, self.edge_importance):
                # h = gcn(h)
                h, _ = gcn(h, self.A * importance)
            shift, scale = split_feature(h, "cross")
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(dim=1)
            if len(shift.shape) == 3:
                shift = shift.unsqueeze(dim=1)
            scale = torch.sigmoid(scale + 2.) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        
        # 5. 应用adapter（如果启用）
        if self.use_adapter:
            z = self.adapter(z)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            if len(z1.shape) == 3:
                z1 = z1.unsqueeze(dim=1)
            if len(z2.shape) == 3:
                z2 = z2.unsqueeze(dim=1)
            h = z1.clone()
            for gcn, importance in zip(self.block, self.edge_importance):
                # h = gcn(h)
                h, _ = gcn(h, self.A * importance)
            # h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(dim=1)
            if len(shift.shape) == 3:
                shift = shift.unsqueeze(dim=1)
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
            self,
            pose_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            edge_importance=False,
            temporal_kernel_size=None,
            strategy='uniform',
            max_hops=8,
            device='cuda:0',
            use_adapter=False,
            adapter_hidden_dim=0,
            adapter_locations=None,
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.use_adapter = use_adapter
        self.adapter_locations = adapter_locations if adapter_locations is not None else []

        self.K = K

        C, T, V = pose_shape
        layer_idx = 0
        for i in range(L):
            if i > 1:
                # 1. Squeeze
                C, T, V = C * 2, T // 2, V
                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, T, V])
            if temporal_kernel_size is None:
                temporal_kernel_size = T // 2 + 1
            # 2. K FlowStep
            for k in range(K):
                last = (k == K - 1)
                first = (k == 0)
                # 检查当前层是否需要添加adapter
                current_use_adapter = self.use_adapter and layer_idx in self.adapter_locations
                
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        temporal_kernel_size=temporal_kernel_size,
                        edge_importance_weighting=edge_importance,
                        last=last,
                        first=first,
                        strategy=strategy,
                        max_hops=max_hops,
                        device=device,
                        use_adapter=current_use_adapter,
                        adapter_hidden_dim=adapter_hidden_dim
                    )
                )
                self.output_shapes.append([-1, C, T, V])
                layer_idx += 1

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        logdet = torch.zeros(z.shape[0]).to(self.device)
        for i, (layer, shape) in enumerate(zip(self.layers, self.output_shapes)):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class STG_NF(nn.Module):
    def __init__(
            self,
            pose_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            learn_top,
            R=0,
            edge_importance=False,
            temporal_kernel_size=None,
            strategy='uniform',
            max_hops=8,
            device='cuda:0',
            use_adapter=False,
            adapter_hidden_dim=0,
            adapter_locations=None,
            use_adapter_pool=False,
            adapter_pool_size=10,
            adapter_pool_threshold=0.1
    ):
        super().__init__()
        self.device = device
        self.use_adapter = use_adapter
        self.use_adapter_pool = use_adapter_pool
        
        # 创建FlowNet
        self.flow = FlowNet(
            pose_shape=pose_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            edge_importance=edge_importance,
            temporal_kernel_size=temporal_kernel_size,
            strategy=strategy,
            max_hops=max_hops,
            device=device,
            use_adapter=use_adapter,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_locations=adapter_locations,
        )
        
        # 创建AdapterPool（如果启用）
        if self.use_adapter_pool:
            C, T, V = pose_shape
            self.adapter_pool = AdapterPool(
                max_size=adapter_pool_size,
                similarity_threshold=adapter_pool_threshold,
                in_channels=C,
                adapter_hidden_dim=adapter_hidden_dim,
                device=device
            )
        self.R = R
        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )
        self.register_buffer(
            "prior_h_normal",
            torch.concat(
                (
                    torch.ones([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]) * self.R,

                    torch.zeros([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                 self.flow.output_shapes[-1][3]]),
                ), dim=0
            ))
        self.register_buffer(
            "prior_h_abnormal",
            torch.concat(
                (
                    torch.ones([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]) * self.R * -1,

                    torch.zeros([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                 self.flow.output_shapes[-1][3]]),
                ), dim=0
            ))

    def prior(self, data, label=None):
        if data is not None:
            if label is not None:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
                h[label == 1] = self.prior_h_normal
                h[label == -1] = self.prior_h_abnormal
            else:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h_normal.repeat(32, 1, 1, 1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        return split_feature(h, "split")

    def forward(self, x=None, z=None, temperature=None, reverse=False, label=None, score=1):
        if reverse:
            return self.reverse_flow(z, temperature)
        else:
            return self.normal_flow(x, label, score)

    def normal_flow(self, x, label, score):
        b, c, t, v = x.shape

        z, objective = self.flow(x, reverse=False)

        mean, logs = self.prior(x, label)
        objective += gaussian_likelihood(mean, logs, z)

        # Full objective - converted to bits per dimension
        nll = (-objective) / (math.log(2.0) * c * t * v)
        # 将nll映射到0-1之间，使用sigmoid函数
        # nll = torch.sigmoid(nll)

        return z, nll

    def reverse_flow(self, z, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
    
    def freeze_main_params(self):
        """冻结主网络参数，只训练adapter"""
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
    
    def unfreeze_main_params(self):
        """解冻主网络参数，允许训练所有参数"""
        for name, param in self.named_parameters():
            param.requires_grad = True
    
    def freeze_adapter_params(self):
        """冻结adapter参数，只训练主网络"""
        for name, param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = False
    
    def unfreeze_adapter_params(self):
        """解冻adapter参数，允许训练adapter"""
        for name, param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
    
    def add_adapter_to_pool(self, adapter=None, target_channels=None, flowstep_params=None):
        """
        向AdapterPool中添加adapter
        """
        if not self.use_adapter_pool:
            raise ValueError("Adapter pool is not enabled. Set use_adapter_pool=True when creating the model.")
        return self.adapter_pool.add_adapter(adapter, target_channels, flowstep_params)
    
    def query_adapter_from_pool(self, query_key, return_index=False):
        """
        从AdapterPool中查询最相似的adapter
        """
        if not self.use_adapter_pool:
            raise ValueError("Adapter pool is not enabled. Set use_adapter_pool=True when creating the model.")
        return self.adapter_pool.query_adapter(query_key, return_index)
    
    def assign_adapters_from_pool(self, query_keys):
        """
        根据查询key从池中分配adapter到指定的FlowStep层
        """
        if not self.use_adapter_pool:
            raise ValueError("Adapter pool is not enabled. Set use_adapter_pool=True when creating the model.")
        
        if not hasattr(self.flow, 'adapter_locations') or not self.flow.adapter_locations:
            raise ValueError("No adapter locations specified.")
        
        if len(query_keys) != len(self.flow.adapter_locations):
            raise ValueError(f"Number of query keys ({len(query_keys)}) must match number of adapter locations ({len(self.flow.adapter_locations)}).")
        
        assigned_adapters = []
        for i, (location, query_key) in enumerate(zip(self.flow.adapter_locations, query_keys)):
            # 从池中查询adapter
            adapter, index = self.adapter_pool.query_adapter(query_key, return_index=True)
            # 将adapter分配给对应的FlowStep
            if location < len(self.flow.layers) and isinstance(self.flow.layers[location], FlowStep):
                self.flow.layers[location].adapter = adapter
                assigned_adapters.append((location, index))
        
        return assigned_adapters
    
    def get_adapter_pool_size(self):
        """
        获取adapter池的大小
        """
        if not self.use_adapter_pool:
            return 0
        return self.adapter_pool.size()


if __name__ == '__main__':
    # 初始化模型参数
    pose_shape = (3, 24, 22)  # C, T, V
    hidden_channels = 0
    K = 8
    L = 1
    actnorm_scale = 1.0
    flow_permutation = "permute"
    flow_coupling = "affine"
    LU_decomposed = True
    learn_top = False
    R = 3
    edge_importance = False
    temporal_kernel_size = None
    strategy = 'uniform'
    max_hops = 8
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 创建模型实例
    model = STG_NF(
        pose_shape=pose_shape,
        hidden_channels=hidden_channels,
        K=K,
        L=L,
        actnorm_scale=actnorm_scale,
        flow_permutation=flow_permutation,
        flow_coupling=flow_coupling,
        LU_decomposed=LU_decomposed,
        learn_top=learn_top,
        R=R,
        edge_importance=edge_importance,
        temporal_kernel_size=temporal_kernel_size,
        strategy=strategy,
        max_hops=max_hops,
        device=device,
        use_adapter=True,
        adapter_locations=[0],
        use_adapter_pool=True,
        adapter_pool_size=3,
    ).to(device)

    print(model)

    for name, param in model.named_parameters():
        print(f"模块名:{name}, 参数形状:{param.shape}")