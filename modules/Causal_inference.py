import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

class hetero_effect_graph(nn.Module):
    def __init__(self, in_channels, out_channels, device, levels=5):
        super(hetero_effect_graph, self).__init__()

        self.device = device

        self.levels = levels + 1

        self.edge_type_mapping = {}
        self.initialize_edge_type_mapping()

        self.conv1 = RGCNConv(in_channels, out_channels, self.levels)
        self.conv2 = RGCNConv(out_channels, out_channels, self.levels)

    def initialize_edge_type_mapping(self):
        j = 0
        for i in range(self.levels + 1):
            edge_type = ('Mole', f'connected__{i}', 'Entity')
            self.edge_type_mapping[edge_type] = j
            j += 1


    def create_hetero_graph(self, emb_entity, emb_mole, entity_mole_weight):
        data = HeteroData()   # 处理异构图数据的类

        data['Entity'].x = emb_entity.squeeze(0)

        data['Mole'].x = emb_mole.squeeze(0)


        weights = np.asarray(entity_mole_weight)
        if weights.shape != (data['Entity'].x.size(0), data['Mole'].x.size(0)):
            raise ValueError("Effect matrix shape does not match node features")
        for i in range(1, self.levels):
            lower = (i - 1) / (self.levels - 1)
            upper = i / (self.levels - 1)
            mask = (weights > lower) & (weights <= upper)
            entity, molecule = np.nonzero(mask)
            data['Mole', f'connected__{i}', 'Entity'].edge_index = torch.tensor(
                np.stack([molecule, entity]), dtype=torch.long)

        return data

    def hetero_to_homo(self, data):
        entity_offset = 0 # 偏移量
        mole_offset = entity_offset + data['Entity'].x.size(0)

        x_all = torch.cat([data['Entity'].x, data['Mole'].x], dim=0)

        edge_index_list = []
        edge_type_list = []

        for i in range(self.levels):
            key = ('Mole', f'connected__{i}', 'Entity') # 起始节点-> 类型 -> 结束节点
            if key in data.edge_types:
                src, dst = data[key].edge_index # edge_index第一行起始索引，赋值给src，结束索引赋值给dst
                edge_index_list.append(torch.stack([src + mole_offset, dst + entity_offset], dim=0))
                edge_type_list.append(torch.full((len(src),), self.edge_type_mapping[key]))

        edge_index = torch.cat(edge_index_list, dim=1).to(self.device)

        edge_type = torch.cat(edge_type_list, dim=0).to(self.device)

        return x_all, edge_index, edge_type

    def forward(self, emb_entity, emb_mole, entity_mole_weights):
        data = self.create_hetero_graph(emb_entity, emb_mole, entity_mole_weights)

        x, edge_index, edge_type = self.hetero_to_homo(data)

        out1 = self.conv1(x, edge_index, edge_type)
        out1 = F.relu(out1)
        out = self.conv2(out1, edge_index, edge_type)

        entity_offset = 0
        mole_offset = entity_offset + data['Entity'].x.size(0)

        out_emb_entity = out[entity_offset:mole_offset]
        out_emb_mole = out[mole_offset:]  # 理论上不需要了

        return out_emb_entity.unsqueeze(0)

class LearnableMaskLayer(nn.Module):
    def __init__(self, emb_dim):
        super(LearnableMaskLayer, self).__init__()
        self.mask_weights = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        return x * self.mask_weights

class CausalWeight(nn.Module):
    def __init__(self, emb_dim, device):
        super(CausalWeight, self).__init__()
        self.device = device
        self.list_weights = nn.ModuleList([LearnableMaskLayer(emb_dim) for _ in range(4)])

    def forward(self, x, causal_graph):
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

        echelon = [[], [], [], []]

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

class homo_relation_graph(nn.Module):
    def __init__(self, emb_dim, device):
        super(homo_relation_graph, self).__init__()
        self.device = device
        self.causal_weight = CausalWeight(emb_dim, device)

    def forward(self, graph, node_features, node_ids):
        data = self.nx_to_pyg(graph, node_features, node_ids)
        x, edge_index = data.x, data.edge_index
        return x

    def nx_to_pyg(self, graph, node_features, node_ids):
        if len(node_ids) != node_features.shape[1] or len(set(node_ids)) != len(node_ids):
            raise ValueError("Node IDs must uniquely match feature rows")
        mapping = {node: i for i, node in enumerate(node_ids)}
        if set(graph.nodes()) != set(node_ids):
            raise ValueError("Cached graph nodes do not match this visit")
        graph = nx.relabel_nodes(graph, mapping)

        node_features2 = self.causal_weight(node_features, graph)

        edge_list = list(graph.edges(data=True))

        edge_index = torch.tensor([[edge[0] for edge in edge_list], [edge[1] for edge in edge_list]]).to(self.device)
        edge_index = edge_index.to(torch.int64)

        return Data(x=node_features2, edge_index=edge_index)
