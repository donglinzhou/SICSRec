import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExpertLayer, self).__init__()
        self.W_g = nn.Linear(input_dim, hidden_dim)
        self.W_p = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        g = torch.sigmoid(self.W_g(x))
        p = self.W_p(x)
        e_i = self.W_o(g * p)
        return e_i


class HierarchicalMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim_id, hidden_dim_content, hidden_dim_id_content):
        super(HierarchicalMoE, self).__init__()

        # Experts for each preference
        self.expert_id = ExpertLayer(input_dim, hidden_dim_id)
        self.expert_content = ExpertLayer(input_dim, hidden_dim_content)
        self.expert_id_content = ExpertLayer(input_dim, hidden_dim_id_content)

        # Gating networks for hierarchical routing
        self.gate_id = nn.Linear(input_dim, 1)
        self.gate_content = nn.Linear(input_dim, 1)
        self.gate_id_content = nn.Linear(input_dim, 1)

    def forward(self, id_pref, content_pref, id_content_pref):

        # Compute expert outputs
        output_id = self.expert_id(id_pref)
        output_content = self.expert_content(content_pref)
        output_id_content = self.expert_id_content(id_content_pref)

        # Compute gate scores
        gate_score_id = torch.sigmoid(self.gate_id(id_pref))
        gate_score_content = torch.sigmoid(self.gate_content(content_pref))
        gate_score_id_content = torch.sigmoid(self.gate_id_content(id_content_pref))

        # Normalize gate scores
        gate_scores = torch.cat([gate_score_id, gate_score_content, gate_score_id_content], dim=-1)
        gate_scores = F.softmax(gate_scores, dim=-1)

        # Hierarchical fusion based on gate scores
        fused_pref = gate_scores[:,  0:1] * output_id + \
                     gate_scores[:, 1:2] * output_content + \
                     gate_scores[:, 2:3] * output_id_content

        return fused_pref #, gate_scores

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_items_emb, neg_items_emb):
        batch_size = pos_items_emb.size(0)

        # Concatenate positive and negative samples
        all_items_emb = torch.cat([pos_items_emb, neg_items_emb], dim=0)  # (2*batch_size, hidden)

        # Calculate similarity matrix
        sim_matrix = torch.matmul(all_items_emb, all_items_emb.t()) / self.temperature  # (2*batch_size, 2*batch_size)

        # Apply log-softmax to get conditional probabilities
        log_probs = nn.functional.log_softmax(sim_matrix, dim=1)

        # Calculate the observed data's conditional probability distribution
        observed_data_probs = log_probs[:batch_size, :batch_size].diag()

        # Calculate the estimated conditional probability distribution
        estimated_data_probs = log_probs[:batch_size, batch_size:].sum(dim=1)

        # Calculate InfoNCE loss
        info_nce_loss = -torch.mean(observed_data_probs - estimated_data_probs)

        return info_nce_loss

class NTXentLoss(torch.nn.Module):  # 不加标签信息的loss
    def __init__(self, temperature=0.1, eps=1e-6):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, out_1, out_2):

        batch_size, seq_length, _ = out_1.size()

        # Flatten the last dimension of embeddings
        out_1 = out_1.reshape(batch_size * seq_length, -1)
        out_2 = out_2.reshape(batch_size * seq_length, -1)

        # out_1: torch.Size([128, 64])
        # out_2: torch.Size([128, 64])
        out_1 = torch.nn.functional.normalize(out_1, p=2, dim=1)
        out_2 = torch.nn.functional.normalize(out_2, p=2, dim=1)

        out = torch.cat([out_1, out_2], dim=0)  # torch.Size([256, 64])
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature).sum(dim=-1)
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / (neg + self.eps)).mean()



class LearnableRoPE(nn.Module):
    def __init__(self, max_position, embedding_size):
        super(LearnableRoPE, self).__init__()
        self.embedding_size = embedding_size
        self.max_position = max_position

        # Define learnable parameters
        self.weights = nn.Parameter(torch.Tensor(max_position, embedding_size))
        self.biases = nn.Parameter(torch.Tensor(max_position, embedding_size))

        # Initialize parameters
        nn.init.normal_(self.weights, mean=0, std=0.1)
        nn.init.normal_(self.biases, mean=0, std=0.1)

    def forward(self, positions):
        # positions: [batch_size, seq_length]

        # Clip positions to avoid out-of-bound errors
        positions = torch.clamp(positions, min=0, max=self.max_position - 1)

        # Gather relative position embeddings
        embeddings = F.embedding(positions, self.weights.long(), sparse=True)
        biases = F.embedding(positions, self.biases.long(), sparse=True)

        # Combine embeddings with biases
        embeddings = embeddings + biases

        return embeddings
