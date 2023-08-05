import torch
import torch.nn.functional as F
from embeddings import EmbBondEncoder,EmbAtomEncoder,PosEncoder
from torch import nn
from NewCore import GINConv,MultiHeadAttention
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set


class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=3, emb_dim=128, n_head=3, residual=False, drop_ratio=0.1,attention=True, JK="last", graph_pooling="mean", data_type='smiles',job_level='node',device=0):
        """GIN Graph Pooling Module
        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表征的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.device= device


        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim,n_head,data_type,drop_ratio=drop_ratio,JK=JK,attention=attention, residual=residual,device=device)

        # Pooling function to generate whole-graph embeddings
        assert job_level in ['graph','node']
        if job_level=='graph':
            if graph_pooling == "sum":
                self.pool = global_add_pool
            elif graph_pooling == "mean":
                self.pool = global_mean_pool
            elif graph_pooling == "max":
                self.pool = global_max_pool
            elif graph_pooling == "attention":
                self.pool = GlobalAttention(gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
            elif graph_pooling == "set2set":
                self.pool = Set2Set(emb_dim, processing_steps=2)
            else:
                raise ValueError("Invalid graph pooling type.")

        elif job_level=='node':
            self.pool = self.skip_it

        # predict
        self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        # 前向传播函数，用于计算模型的输出

        # 节点嵌入层
        h_node = self.gnn_node(batched_data)

        # 池化层
        h_graph = self.pool(h_node, batched_data.batch)

        # 全图预测层
        output = self.graph_pred_linear(h_graph)
        return output

        # # 如果是训练模式，则直接输出output
        # if self.training:
        #     return output
        # # 如果是测试模式，则将output的值限制在0-50之间，然后输出
        # else:
        #     # At inference time, relu is applied to output to ensure positivity
        #     # 因为预测目标的取值范围就在 (0, 50] 内
        #     return torch.clamp(output, min=0, max=1000)


    def skip_it(self,input,batch):
        return input


# GNN to generate node embedding 第二层
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim,n_head, data_type, drop_ratio=0.1, JK="last", residual=False,attention=True,device=0):
        """GIN Node Embedding Module"""

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim=emb_dim
        self.attention=attention
        if data_type=='smiles':
            self.n_features=9
        elif data_type=='crystal':
            self.n_features = 1

        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = EmbAtomEncoder(emb_dim,self.n_features) #构建原子表征embedding
        # self.pos_encoder =PosEncoder(emb_dim)

        # List of GNNs
        self.gnnconvs = torch.nn.ModuleList()
        self.attentionconvs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.gnnconvs.append(GINConv(emb_dim,device=device))
            self.attentionconvs.append(MultiHeadAttention(n_head, emb_dim,dropout=drop_ratio))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index,bond_length, edge_attr,pos,num_nodes = batched_data.x, batched_data.edge_index,batched_data.bond_length, batched_data.edge_attr,batched_data.new_pos,batched_data.num_nodes


        # computing input node embedding
        h_list = [self.atom_encoder(x,pos)]  # 先将类别型原子属性转化为原子表征
        for layer in range(self.num_layers):
            h = self.gnnconvs[layer](h_list[layer], edge_index, bond_length=bond_length,edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
            if self.attention:

                h = h.view(len(batched_data), -1, self.emb_dim)
                # print(h.shape)
                h,*_ = self.attentionconvs[layer](h,h,h)
                # print(num_nodes)
                h= h.view(num_nodes,self.emb_dim)

            if self.residual:
                h += h_list[layer]

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)



            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation

