import torch
import torch.nn as nn
# import torch.nn.functional as F

Tensor = torch.Tensor


class GIM(nn.Module):
    def __init__(self, gnn, out_feats, num_nodes, num_edges, num_inds=None, num_cls=None, num_samples=None, net_type='ind', use_init_struc=False, init_struc=None, structure_learning=False, batch_size=32):
        super(GIM, self).__init__()
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

    def gumbel_softkmax(self, logits: Tensor, tau: float = 0.5, k: int = 20, hard: bool = False) -> Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = (torch.sigmoid(gumbels)/2+torch.sigmoid(gumbels.transpose(1, 2))/2).view(logits.shape[0], -1).view_as(logits) #.softmax(-1)

        k = (y_soft-torch.diag_embed(torch.diagonal(y_soft, dim1=-2, dim2=-1))/2).sum(dim=(-2, -1)).int()
        if hard:
            if type(k) == int:
                # Straight through.
                upper_triangular = torch.triu(y_soft)
                y_soft_tmp = upper_triangular.view(logits.shape[0], -1)
                index = y_soft_tmp.topk(k)[1]
                y_hard = torch.zeros_like(logits.view_as(y_soft_tmp), memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            else:

                B, N, M = y_soft.shape
                device = y_soft.device

                # 展平每个矩阵
                batch_flat = y_soft.view(B, -1)  # 形状 (B, N*M)

                # 对每个矩阵排序
                sorted_values, sorted_indices = torch.sort(batch_flat, dim=1, descending=True)  # 形状 (B, N*M)

                # 生成每个矩阵的 k 索引范围
                range_indices = torch.arange(N*M, device=device).unsqueeze(0).expand(B, -1)  # 形状 (B, N*M)
                k_expand = k.unsqueeze(1)  # 形状 (B, 1)
                mask = range_indices < k_expand  # 形状 (B, N*M)

                # 创建一个全零的张量来存储结果
                output_flat = torch.zeros_like(batch_flat)  # 形状 (B, N*M)

                # 只保留每个矩阵的 top-k 元素，其余置零
                topk_values = sorted_values * mask.float()  # 形状 (B, N*M)

                # 获取排序后的索引，并在原始位置上放置 top-k 元素
                output_flat.scatter_(1, sorted_indices, 1.0)

                # 将结果重新 reshape 回原始矩阵形状
                # y_hard = output_flat.view(B, N, M)



                # index = y_soft_tmp.topk(k)[1]

                # y_hard = torch.zeros_like(logits.view_as(y_soft_tmp), memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
                y_hard = output_flat.view_as(logits)
            y_hard = (y_hard + y_hard.transpose(1, 2)).clamp(0, 1)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret, y_soft