import torch
import torch.nn as nn
import torch.nn.functional as F
F.gumbel_softmax

Tensor = torch.Tensor


class CGSL(nn.Module):
    def __init__(self, gnn, out_feats, num_nodes, num_edges, num_inds=None, num_cls=None, num_samples=None, net_type='ind', use_init_struc=False, init_struc=None, structure_learning=False, batch_size=32):
        super(CGSL, self).__init__()
        self.num_edges = int(num_edges / 100 * num_nodes * (num_nodes - 1) / 2)
        self.gnn = gnn
        self.net_type = net_type
        net_type_dict = {'ind': num_inds, 'cls': num_cls, 'sample': num_samples, 'group': 1}
        self.num_nets = net_type_dict[self.net_type]
        self.use_init_struc = use_init_struc
        self.train_flag = structure_learning
        if self.use_init_struc == True:
            self.nets = nn.Parameter(torch.tensor(init_struc, dtype=torch.float32))
        else:
            self.nets = nn.Parameter(torch.ones(self.num_nets, num_nodes, num_nodes))
        self.batch_size = batch_size
        self.linear = nn.Linear(out_feats, num_cls)

    def forward(self, data, net_index):
        x = data

        if self.net_type == 'group':
            selected_nets = self.nets.repeat([x.shape[0], 1, 1])
        else:
            selected_nets = self.nets[net_index]
        # Create a network structure based on the masked_features
        if self.train_flag == False or self.net_type == 'group':
            adj_matrix = selected_nets
        else:
            # network_structure = torch.squeeze(torch.sigmoid(selected_nets))
            network_structure = torch.squeeze(selected_nets)

            # Convert network_structure to adjacency matrix
            adj_matrix = self.gumbel_softkmax(network_structure, k=self.num_edges, hard=True)

        embeddings = self.gnn(x, adj_matrix)

        output = self.linear(embeddings)

        return output, embeddings, adj_matrix

    def gumbel_softkmax(self, logits: Tensor, tau: float = 1, k = 20, hard: bool = False) -> Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = ((gumbels + gumbels.transpose(1, 2))/2).view(logits.shape[0], -1).softmax(-1).view_as(logits)

        if hard:
            # Straight through.
            upper_triangular = torch.triu(y_soft)
            y_soft_tmp = upper_triangular.view(logits.shape[0], -1)
            if type(k) == int:
                index = y_soft_tmp.topk(k)[1]
            # TODO: k is a tensor
            else:
                top_k_results = []
                assert k.shape[0] == y_soft_tmp.shape[0]
                sorted_indices = torch.argsort(y_soft_tmp, axis=1)
                for row, k in zip(sorted_indices, y_soft_tmp):
                    # 选取每行中最大的 K 个元素的索引
                    top_k_indices = row[-k:]
                    # 使用高级索引从原数组中提取元素
                    top_k_elements = y_soft_tmp[torch.arange(y_soft_tmp.shape[0]), top_k_indices]
                    top_k_results.append(top_k_elements)
                index = torch.array(top_k_results)

            y_hard = torch.zeros_like(logits.view_as(y_soft_tmp), memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            y_hard = y_hard.view_as(logits)
            y_hard = y_hard + y_hard.transpose(1, 2) - torch.diag_embed(torch.diagonal(y_hard, dim1=-2, dim2=-1))
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret