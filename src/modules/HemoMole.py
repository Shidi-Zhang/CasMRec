from .SetTransformer import SAB
from .gnn import GNNGraph

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .Causal_inference import hetero_effect_graph, homo_relation_graph

class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_diag, num_proc, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph
        diag_med_high = casual_graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_med_low = casual_graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_med_high = casual_graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_med_low = casual_graph.get_threshold_effect(0.90, "Proc", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([diag_med_high, proc_med_high]))  # 选用的97%
        self.c1_low_limit = nn.Parameter(torch.tensor([diag_med_low, proc_med_low]))  # 选用的90%
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, pre_prob, diags, procs):
        reviewed_prob = pre_prob.clone()

        for m in range(self.num_med):
            max_cdm = 0.0
            max_cpm = 0.0
            for d in diags:
                cdm = self.c1.get_effect(d, m, "Diag", "Med")
                max_cdm = max(max_cdm, cdm)
            for p in procs:
                cpm = self.c1.get_effect(p, m, "Proc", "Med")
                max_cpm = max(max_cpm, cpm)

            if max_cdm < self.c1_low_limit[0] and max_cpm < self.c1_low_limit[1]:
                reviewed_prob[0, m] -= self.c1_minus_weight
            elif max_cdm > self.c1_high_limit[0] or max_cpm > self.c1_high_limit[1]:
                reviewed_prob[0, m] += self.c1_plus_weight

        return reviewed_prob
class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)  # [131, 491]

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        # print(Attn[0])
        # print(mask[0])
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O

class HemoMoleModel(torch.nn.Module):
    def __init__(
            self, global_para, substruct_para,causal_graph, emb_dim, voc_size,
            substruct_num, global_dim, substruct_dim, mole_relevance, use_embedding=True,
            device=torch.device('cpu'), dropout=0.5, *args, **kwargs
    ):
        super(HemoMoleModel, self).__init__(*args, **kwargs)
        # *args用于接收任意数量的位置参数，将这些参数打包成一个元组（tuple）传递给函数。
        # **kwargs用于接收任意数量的关键字参数，将这些参数打包成一个字典（dictionary）传递给函数。
        self.device = device
        self.use_embedding = use_embedding  # 是否使用嵌入

        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )  # 大小为substruct_num行、emb_dim列的全零张量 492*64
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GNNGraph(**global_para)
        self.emb_dim = emb_dim

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim),
            torch.nn.Embedding(voc_size[3], emb_dim)
        ])
        self.homo_graph = torch.nn.ModuleList([
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device)
        ])

        self.hetero_graph = torch.nn.ModuleList([
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])

        self.causal_graph = causal_graph
        self.mole_relevance = mole_relevance
        self.review = CausaltyReview(self.causal_graph, voc_size[0], voc_size[1], voc_size[2])

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.mole_med_relevance = nn.Parameter(torch.tensor(mole_relevance[2],dtype=torch.float))
        self.sab = SAB(substruct_dim, substruct_dim, 2,
                       use_ln=True)  # 输入维度为substruct_dim，输出维度为substruct_dim，头数为2，使用了LN归一化
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )  # 定义了一个查询层，包括ReLU激活函数和线性变换，输入维度为emb_dim * 4，输出维度为emb_dim。256->64
        self.query1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 6, emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim,
                                              substruct_num)  # 输入维度为emb_dim的特征映射到输出维度为substruct_num的空间中 64 -> 492
        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )  # 这段代码定义了一个聚合层，它将全局特征和局部特征进行聚合，聚合方式是通过相似度矩阵进行加权平均，然后通过SAB（可分解自注意力）进行处理。
        score_extractor = [
            torch.nn.Linear(substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, 1)
        ]  # 从子结构特征中提取相关信息
        self.score_extractor = torch.nn.Sequential(*score_extractor)  # 将这些层组合起来，形成一个序列模型
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        #if self.use_embedding:
         #   torch.nn.init.xavier_uniform_(self.substruct_emb)

    def med_embedding(self, idx, emb_mole):
        relevance = self.mole_med_relevance[idx, :].to(self.device)

        mask = relevance != 0
        relevance_masked = relevance.masked_fill(~mask, -float('inf'))
        relevance_normalized = F.softmax(relevance_masked, dim=1)
        emb_med1 = torch.matmul(relevance_normalized, emb_mole.squeeze(0))

        return emb_med1.unsqueeze(0)


    def forward(
            self, substruct_data, mol_data, patient_data,
            ddi_mask_H, tensor_ddi_adj, average_projection
    ):
        seq1, seq2, seq3= [], [], []  #用于序列处理
        for adm_id, adm in enumerate(patient_data):
            # 获取分子嵌入
            num_moles = self.embeddings[3].num_embeddings
            idx_mole = torch.arange(num_moles).to(self.device)
            emb_mole = self.embeddings[3](idx_mole).unsqueeze(0) #[1,284,64]

            Idx_diag = torch.LongTensor([adm[0]]).to(self.device)
            Idx_proc = torch.LongTensor([adm[1]]).to(self.device)
            emb_diag = self.rnn_dropout(self.embeddings[0](Idx_diag))  # 获取诊断的嵌入表示
            emb_proc = self.rnn_dropout(self.embeddings[1](Idx_proc))  # 获取处理的嵌入表示
            #print("emb_diag",emb_diag.shape,"emb_proc",emb_proc.shape)


            relevance_diag = self.mole_relevance[0][adm[0], :]
            emb_diag1 = self.hetero_graph[0](emb_diag, emb_mole, relevance_diag)

            relevance_proc = self.mole_relevance[1][adm[1], :]
            emb_proc1 = self.hetero_graph[1](emb_proc, emb_mole, relevance_proc)

            graph_diag = self.causal_graph.get_graph(adm[3], "Diag")
            graph_proc = self.causal_graph.get_graph(adm[3], "Proc")
            emb_diag2 = self.homo_graph[0](graph_diag, emb_diag1)
            emb_proc2 = self.homo_graph[1](graph_proc, emb_proc1)

            if adm == patient_data[0]:
                emb_med2 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                emb_med1 = self.rnn_dropout(self.med_embedding(adm_last[2], emb_mole))

                med_graph = self.causal_graph.get_graph(adm_last[3], "Med")
                emb_med2 = self.homo_graph[2](med_graph, emb_med1)

            seq1.append(torch.sum(emb_diag2, keepdim=True, dim=1))
            seq2.append(torch.sum(emb_proc2, keepdim=True, dim=1))
            seq3.append(torch.sum(emb_med2, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)  # 将诊断的嵌入表示连接起来
        seq2 = torch.cat(seq2, dim=1)
        seq3 = torch.cat(seq3, dim=1)

        output1, hidden1 = self.seq_encoders[0](seq1)  # 使用GRU进行编码
        output2, hidden2 = self.seq_encoders[1](seq2)
        output3, hidden3 = self.seq_encoders[2](seq3)


        seq_repr = torch.cat([hidden1, hidden2, hidden3], dim=-1)
        last_repr = torch.cat([output1[:, -1], output2[:, -1], output3[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        seq_repr1 = torch.cat([hidden1, hidden2], dim=-1)
        last_repr1 = torch.cat([output1[:, -1], output2[:, -1]], dim=-1)
        patient_repr1 = torch.cat([seq_repr1.flatten(), last_repr1.flatten()])

        query = self.query(patient_repr1)  # 查询
        substruct_weight = torch.sigmoid(self.substruct_rela(query))  # 子结构权重

        global_embeddings = self.global_encoder(**mol_data)  # 全局特征
        global_embeddings = torch.mm(average_projection, global_embeddings)  # 使用平均投影矩阵对全局嵌入进行变换

        substruct_embeddings = self.sab(
            self.substruct_emb.unsqueeze(0)
        ).squeeze(0)  # 子结构特征 ES*
        molecule_embeddings = self.aggregator(
            global_embeddings, substruct_embeddings,
            substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
        )  # 将全局嵌入、子结构嵌入和子结构权重合并为分子嵌入[131,64]

        """"for adm_id, adm in enumerate(patient_data):
            if adm == patient_data[0]:
                emb_med2 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                emb_med1 = self.rnn_dropout(self.med_embedding(adm_last[2], molecule_embeddings))

                med_graph = self.causal_graph.get_graph(adm_last[3], "Med")
                emb_med2 = self.homo_graph[2](med_graph, emb_med1)
            seq3.append(torch.sum(emb_med2, keepdim=True, dim=1))
        seq3 = torch.cat(seq3, dim=1)
        output3, hidden3 = self.seq_encoders[2](seq3)
        seq_repr = torch.cat([hidden1, hidden2, hidden3], dim=-1)
        last_repr = torch.cat([output1[:, -1], output2[:, -1], output3[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])"""

        #计算得分和负样本损失
        score = self.score_extractor(molecule_embeddings).t()  # 从子结构特征中提取相关信息

        score = self.review(score, patient_data[-1][0], patient_data[-1][1])

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()  # 负样本损失
        return score, batch_neg

        # 输出药物推荐的得分和负样本损失，用于训练过程中的负采样正则化
