import torch
from torch import nn
import math
from Dataset_Producer.Smiles_producer.Present import get_bond_feature_dims,get_atom_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

# class PositionalEncoding(nn.Module):
#     """
#     Implements the sinusoidal positional encoding for
#     non-recurrent neural networks.
#
#     Implementation based on "Attention Is All You Need"
#     :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
#
#     Args:
#        dropout_prob (float): dropout parameter
#        dim (int): embedding size
#     """
#
#     def __init__(self, num_embeddings:list, embedding_dim, dropout_prob=0., padding_idx=0):
#         super(PositionalEncoding, self).__init__()
#
#         # pe = torch.zeros(max_len, dim)
#         # position = torch.arange(0, max_len).unsqueeze(1)
#         # div_term = torch.exp((torch.arange(0, dim, 2) *
#         #                      -(math.log(10000.0) / dim)).float())
#         # pe[:, 0::2] = torch.sin(position.float() * div_term)
#         # pe[:, 1::2] = torch.cos(position.float() * div_term)
#         # pe = pe.unsqueeze(0)
#
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.embedding_list = nn.ModuleList()
#         self.weight=[]
#
#         for i,dim in enumerate(num_embeddings):
#             embedding = nn.Embedding(dim, embedding_dim, padding_idx=padding_idx)
#             # torch.nn.init.xavier_uniform_(embbedding.weight.data)#初始化函数，会吧padding的0给抹掉
#             self.embedding_list.append(embedding)
#             self.weight.append(embedding.weight)
#         # self.register_buffer('pe', pe)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.dim = embedding_dim
#
#     def forward(self, x, pos=None):
#         xx=0
#         #xx.shape[batchsize,length,feature_size]
#         for i in range(x.shape[2]):
#             print(i)
#             xx += self.embedding_list[i](x[:,:,i])
#         # x.shape = [2, 29, 128] - -[batchsize, maxlen, d_model]
#         xx = xx * math.sqrt(self.dim)
#         if pos is None:
#             pass
#         else:
#             print('xx',xx.shape)
#             xx = xx + self.pe(pos)
#         xx = self.dropout(xx)
#         return xx

class PosEncoder(torch.nn.Module):
    def __init__(self,emb_dim,device=0):
        super(PosEncoder, self).__init__()
        self.emb_dim=emb_dim
        self.device=device

    def forward(self,pos):
        pee=torch.zeros(pos.shape[0], self.emb_dim).to(self.device)

        for i in range(0,pos.shape[1]):
            pe = torch.zeros(pos.shape[0], self.emb_dim).to(self.device)
            # print('pe1',pe)

            div_term = torch.exp((torch.arange(0, self.emb_dim, 2) *
                                 -(math.log(10000.0) / self.emb_dim)).float()).to(self.device)
            pe[:,0::2] = torch.sin(pos[:,i].unsqueeze(1).float() * div_term)
            pe[:,1::2] = torch.cos(pos[:,i].unsqueeze(1).float() * div_term)
            # print('pee', pee)
            pee += pe
        # print(pee.shape)
        return pee




# def PosEncoder(emb_dim,pos):
#         #此处pos应该升过维与x对应
#         pee=0
#
#         for i in range(0,pos.shape[1]):
#             pe = torch.zeros(pos.shape[0], emb_dim)
#             # print('pe1',pe)
#
#             div_term = torch.exp((torch.arange(0, emb_dim, 2) *
#                                  -(math.log(10000.0) / emb_dim)).float())
#             pe[:,0::2] = torch.sin(pos[:,i].unsqueeze(1).float() * div_term)
#             pe[:,1::2] = torch.cos(pos[:,i].unsqueeze(1).float() * div_term)
#             # print('pee', pee)
#             pee += pe
#         print(pee.shape)
#         return pee


class EmbAtomEncoder(torch.nn.Module):
    """该类用于对原子属性做嵌入。
    记`N`为原子属性的维度，则原子属性表示为`[x1, x2, ..., xi, xN]`，其中任意的一维度`xi`都是类别型数据。full_atom_feature_dims[i]存储了原子属性`xi`的类别数量。
    该类将任意的原子属性`[x1, x2, ..., xi, xN]`转换为原子的嵌入`x_embedding`（维度为emb_dim）。
    """
    def __init__(self, emb_dim,n_features): # 初始化函数，接受一个emb_dim参数
        super(EmbAtomEncoder, self).__init__()

        # 创建一个ModuleList对象，用于存储不同属性的embedding
        self.atom_embedding_list = torch.nn.ModuleList()
        self.pos_encoder = PosEncoder(emb_dim)
        self.n_features=n_features

        # 遍历full_atom_feature_dims列表中的每个元素，并为每个属性创建一个embedding
        for i in range(0,n_features):
            emb = torch.nn.Embedding(full_atom_feature_dims[i], emb_dim)  # 不同维度的属性用不同的Embedding方法
            torch.nn.init.xavier_uniform_(emb.weight.data)  # 初始化权重参数
            self.atom_embedding_list.append(emb)  # 将embedding添加到ModuleList中

    # 前向传播函数，接受输入张量x
    def forward(self, x,pos):
        x_embedding = 0
        x_embedding +=self.pos_encoder(pos)
        # 对于x的第二个维度，即对于每个属性，计算其embedding，并累加
        for i in range(self.n_features):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class EmbBondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(EmbBondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):

        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding

