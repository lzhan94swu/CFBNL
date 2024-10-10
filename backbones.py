import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGINConv, DenseGATConv, DenseSAGEConv, MLP

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = DenseGCNConv(in_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj, add_loop=False))
        x = F.relu(self.conv2(x, adj, add_loop=False))
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = DenseSAGEConv(in_dim, hidden_dim)
        self.conv2 = DenseSAGEConv(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(GIN, self).__init__()
        nn1 = MLP([in_dim, hidden_dim], norm=None)
        nn2 = MLP([hidden_dim, out_dim], norm=None)
        # nn1 = Linear(in_dim, hidden_dim)
        # nn2 = Linear(hidden_dim, out_dim)

        self.conv1 = DenseGINConv(nn1)
        self.conv2 = DenseGINConv(nn2)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj, add_loop=False))
        x = F.relu(self.conv2(x, adj, add_loop=False))
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(GAT, self).__init__()
        self.conv1 = DenseGATConv(in_dim, hidden_dim, heads=8, dropout=0.5)

        self.conv2 = DenseGATConv(hidden_dim * 8, out_dim, heads=1,
                             concat=False, dropout=0.5)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj):
        x = F.leaky_relu(self.conv1(x, adj, add_loop=False))
        x = F.leaky_relu(self.conv2(x, adj, add_loop=False))
        return x