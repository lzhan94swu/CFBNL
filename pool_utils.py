# Code from https://github.com/juho-lee/set_transformer
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.experimental import (
    disable_dynamic_shapes,
    is_experimental_mode_enabled,
)
from torch_geometric.utils import cumsum, scatter

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

@disable_dynamic_shapes(required_args=['batch_size', 'max_num_nodes'])
def to_dense_batch(
    x: Tensor,
    batch: Optional[Tensor] = None,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)

    Examples:

        >>> x = torch.arange(12).view(6, 2)
        >>> x
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])

        >>> out, mask = to_dense_batch(x)
        >>> mask
        tensor([[True, True, True, True, True, True]])

        >>> batch = torch.tensor([0, 0, 1, 2, 2, 2])
        >>> out, mask = to_dense_batch(x, batch)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11]]])
        >>> mask
        tensor([[ True,  True, False],
                [ True, False, False],
                [ True,  True,  True]])

        >>> out, mask = to_dense_batch(x, batch, max_num_nodes=4)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11],
                [ 0,  0]]])

        >>> mask
        tensor([[ True,  True, False, False],
                [ True, False, False, False],
                [ True,  True,  True, False]])
    """
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = cumsum(num_nodes)

    filter_nodes = False
    dynamic_shapes_disabled = is_experimental_mode_enabled(
        'disable_dynamic_shapes')

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif not dynamic_shapes_disabled and num_nodes.max() > max_num_nodes:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    if filter_nodes:
        mask = tmp < max_num_nodes
        x, idx = x[mask], idx[mask]

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = torch.as_tensor(fill_value, device=x.device)
    out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask