import torch
import torch.nn.functional as F
from embeddings import EmbBondEncoder,EmbAtomEncoder,PosEncoder
from torch import nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree



class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "mean")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = EmbBondEncoder(emb_dim = emb_dim)
        self.pos_encoder = PosEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, bond_length, edge_attr=None):
        if edge_attr==None:
            edge_attr_embedding = self.pos_encoder(bond_length)
            # print(edge_attr_embedding.shape)
        else:
            edge_attr_embedding = self.bond_encoder(edge_attr) # 先将类别型边属性转换为边表征
            edge_attr_embedding += self.pos_encoder(bond_length)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_attr_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out#得看看维度


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()


        self.n_head = n_head
        self.d_head = d_model // n_head
        # 三个线性层做矩阵乘法生成q, k, v.
        self.w_qs = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.fc = nn.Linear(n_head * self.d_head, d_model, bias=False)
        # ScaledDotProductAttention见下方
        self.attention = ScaledDotProductAttention(temperature=self.d_head ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_head, self.d_head, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # b: batch_size, lq: translation task的seq长度, n: head数, dv: embedding vector length
        # Separate different heads: b x lq x n x dv.
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # project & reshape
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # view只能用在contiguous的variable上
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # add & norm
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q x k^T
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))


        # dim=-1表示对最后一维softmax
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn