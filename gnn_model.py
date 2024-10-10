import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool

from backbones import *
from pools import *


class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_node, conv='gcn', read_out_type='mean'):
        super(GNNModel, self).__init__()
        self.conv = conv
        if self.conv == 'gcn':
            self.gnn = GCN(in_feats, out_feats, hidden_feats)
        elif self.conv == 'sage':
            self.gnn = GraphSAGE(in_feats, out_feats, hidden_feats)
        elif self.conv == 'gat':
            self.gnn = GAT(in_feats, out_feats, hidden_feats)
        elif self.conv == 'gin':
            self.gnn = GIN(in_feats, out_feats, hidden_feats)
        else:
            print('conv type error')
            exit(1)

        self.read_out_type = read_out_type
        if self.read_out_type == 'mean':
            self.readout = GlobalMeanPool()
        if self.read_out_type == 'max':
            self.readout = GlobalMaxPool()
        if self.read_out_type == 'add':
            self.readout = GlobalAddPool()

        if self.read_out_type == 'dense':
            self.readout = torch.nn.Sequential(
                Linear(in_features=out_feats, out_features=out_feats),
                BatchNorm1d(out_feats),
                ReLU(),
                Linear(in_features=out_feats, out_features=out_feats),
                BatchNorm1d(out_feats),
                ReLU(),
                Dropout(p=0.4)
            )
        if self.read_out_type == 'trans':
            self.readout = SetTransformer(dim_input=out_feats, num_outputs=1, dim_output=out_feats)

        if self.read_out_type == 'dense':
            self.lin1 = Linear(num_node * out_feats, out_feats)
        else:
            self.lin1 = Linear(out_feats, out_feats)
        self.lin2 = Linear(out_feats, out_feats)

    def forward(self, x, adj):
        x = self.gnn(x, adj)
        if self.read_out_type == 'dense':
            x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.readout(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return x