import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

# 实现异构图神经网络模型，用于处理异构图数据结构和进行神经网络计算
class hetero_effect_graph(nn.Module):
    def __init__(self, in_channels, out_channels, device, levels=5):
        super(hetero_effect_graph, self).__init__()

        self.device = device

        # 等级数量，用于划分不同权重级别的边 最后加1代表生成一种虚拟的边
        self.levels = levels + 1

        # 边类型映射字典
        self.edge_type_mapping = {}
        self.initialize_edge_type_mapping()

        # 定义两个RGCN卷积层
        self.conv1 = RGCNConv(in_channels, out_channels, self.levels)
        self.conv2 = RGCNConv(out_channels, out_channels, self.levels)

    def initialize_edge_type_mapping(self):
        # 分配整数值给每种边类型
        j = 0
        for i in range(self.levels + 1):
            edge_type = ('Mole', f'connected__{i}', 'Entity')
            self.edge_type_mapping[edge_type] = j
            j += 1


    def create_hetero_graph(self, emb_entity, emb_mole, entity_mole_weight):
        # 创建异构图数据结构
        data = HeteroData()   # 处理异构图数据的类

        # 分配节点嵌入--实体节点和分子节点
        data['Entity'].x = emb_entity.squeeze(0)

        data['Mole'].x = emb_mole.squeeze(0)


        # 如果全部是0向量不用分层
        if np.all(entity_mole_weight == 0):
            src = torch.zeros(entity_mole_weight.size, dtype=torch.int64)
            dst = torch.arange(0, entity_mole_weight.size, dtype=torch.int64)  # 从0到entity_mole_weight大小
            edge_index = torch.stack([src, dst]) # 堆叠，表示边的索引
            data['Mole', f'connected__{0}', 'Entity'].edge_index = edge_index  # 将边索引赋值给异构图
            print('如果全部是0向量不用分层')
        else:
            # 根据权重为关系分配不同的关系类型
            for i in range(1, self.levels):
                # 创建一个布尔掩码，筛选出权重在 (i / self.levels) 和 ((i + 1) / self.levels) 之间的元素。
                mask = (entity_mole_weight > (i / self.levels)) & \
                       (entity_mole_weight <= ((i + 1) / self.levels))
                edge_index = torch.from_numpy(np.vstack(mask.nonzero())) # 用于将布尔掩码数组mask中为True的元素的索引按行堆叠成一个二维数组
                # edge_index张量的形状通常为 (2, num_edges)，是一个包含两行的二维张量，其中每列代表一条边的起始节点和结束节点索引

                if edge_index.size(0) > 0:
                    # 不需要具体的权重，知道属于第几类边就可以了
                    edge_index = edge_index.flip([0])  # 用于沿着指定维度翻转张量的顺序，将行列索引互换
                    data['Mole', f'connected__{i}', 'Entity'].edge_index = edge_index

        return data

    # 将异构图数据转换为同构图数据
    def hetero_to_homo(self, data):
        # 统一编码所有节点
        entity_offset = 0 # 偏移量
        mole_offset = entity_offset + data['Entity'].x.size(0)
        #print(data['Entity'].x.shape,data['Mole'].x.shape) #.x表示获得Entity节点的特征张量，整体表示获取节点特征张量的行数，即特征张量第一个维度的大小

        # 合并所有节点特征，x_all是所有节点的嵌入，将实体节点和分子节点的特征矩阵在第0维上进行拼接
        x_all = torch.cat([data['Entity'].x, data['Mole'].x], dim=0)

        # 创建整张图的edge_index和edge_type
        edge_index_list = []
        edge_type_list = []

        # range+1为了适配虚拟类
        for i in range(self.levels):
            key = ('Mole', f'connected__{i}', 'Entity') # 起始节点-> 类型 -> 结束节点
            if key in data.edge_types:
                src, dst = data[key].edge_index # edge_index第一行起始索引，赋值给src，结束索引赋值给dst
                edge_index_list.append(torch.stack([src + mole_offset, dst + entity_offset], dim=0))
                # 将起始节点索引 src 加上 mole_offset，结束节点索引 dst 加上 entity_offset，然后沿着行维度（dim=0）将它们堆叠成一个新的张量，并将这个张量添加到 edge_index_list 列表中
                edge_type_list.append(torch.full((len(src),), self.edge_type_mapping[key]))
                # 将长度为 len(src) 的张量填充为 self.edge_type_mapping[key] 的值，然后将这个张量添加到 edge_type_list 列表中。

        # Concatenate edge_index from different edge types
        edge_index = torch.cat(edge_index_list, dim=1).to(self.device)

        # Concatenate edge_type from different edge types
        edge_type = torch.cat(edge_type_list, dim=0).to(self.device)

        return x_all, edge_index, edge_type

    def forward(self, emb_entity, emb_mole, entity_mole_weights):
        # 创建异构图
        data = self.create_hetero_graph(emb_entity, emb_mole, entity_mole_weights)

        # 从异构图转换到同构图
        x, edge_index, edge_type = self.hetero_to_homo(data)

        # 卷积
        out1 = self.conv1(x, edge_index, edge_type)
        out1 = F.relu(out1)
        out = self.conv2(out1, edge_index, edge_type)

        # 根据偏移量切割张量，分解出每种类型的嵌入
        entity_offset = 0
        mole_offset = entity_offset + data['Entity'].x.size(0)

        out_emb_entity = out[entity_offset:mole_offset]
        out_emb_mole = out[mole_offset:]  # 理论上不需要了

        return out_emb_entity.unsqueeze(0)

class LearnableMaskLayer(nn.Module):
    def __init__(self, emb_dim):
        super(LearnableMaskLayer, self).__init__()
        # 创建一个与输入相同形状的权重，初始值设为1
        self.mask_weights = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # 将输入与mask权重相乘
        return x * self.mask_weights

# 根据因果图对输入进行加权处理，获得各节点的权重
class CausalWeight(nn.Module):
    def __init__(self, emb_dim, device):
        super(CausalWeight, self).__init__()
        self.device = device
        self.list_weights = nn.ModuleList([LearnableMaskLayer(emb_dim) for _ in range(4)])

    def forward(self, x, causal_graph):
        # 遍历每个列表
        echelon = self.node_classify(causal_graph)
        x1 = torch.zeros_like(x)
        for i, node_list in enumerate(echelon):
            for node in node_list:
                x1[0, node, :] += self.list_weights[i](x[0, node, :])
        return x1

    def node_classify(self, causal_graph):
        """初始化四个类别的列表 从因到果
        0无入度有出度 因节点
        1无入度无出度 孤儿节点
        2有入度有出度 中间节点
        3有入度无出度 果节点"""

        # 根据node_type初始化列表
        echelon = [[], [], [], []]

        # 对每个节点进行分类
        for node in causal_graph.nodes():
            in_degree = causal_graph.in_degree(node)
            out_degree = causal_graph.out_degree(node)

            if in_degree == 0 and out_degree == 0:
                echelon[1].append(node)
            elif in_degree > 0 and out_degree == 0:
                echelon[3].append(node)
            elif in_degree == 0 and out_degree > 0:
                echelon[0].append(node)
            else:
                echelon[2].append(node)

        return echelon

#根据因果图对输入数据进行加权处理，并将处理后的数据转换为 PyTorch Geometric 数据对象。
class homo_relation_graph(nn.Module):
    def __init__(self, emb_dim, device):
        super(homo_relation_graph, self).__init__()
        self.device = device
        self.causal_weight = CausalWeight(emb_dim, device)

    def forward(self, graph, node_features):
        data = self.nx_to_pyg(graph, node_features)
        x, edge_index = data.x, data.edge_index
        return x

    # 将输入的图数据和节点特征转换为 PyTorch Geometric 数据对象
    def nx_to_pyg(self, graph, node_features):
        # 将字符串节点转换为整数节点
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)

        # 根据因果性重新分配各个节点权重
        node_features2 = self.causal_weight(node_features, graph)

        # 边list
        edge_list = list(graph.edges(data=True))

        # 获得边
        edge_index = torch.tensor([[edge[0] for edge in edge_list], [edge[1] for edge in edge_list]]).to(self.device)
        edge_index = edge_index.to(torch.int64)

        return Data(x=node_features2, edge_index=edge_index)