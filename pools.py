import torch.nn as nn

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return torch.squeeze(self.dec(self.enc(X)))

class GlobalMeanPool(nn.Module):
    def forward(self, x):
        # x 的形状是 (batch, n, dim_feat)
        return torch.mean(x, dim=1)

class GlobalMaxPool(nn.Module):
    def forward(self, x):
        # x 的形状是 (batch, n, dim_feat)
        return torch.max(x, dim=1)[0]

class GlobalAddPool(nn.Module):
    def forward(self, x):
        # x 的形状是 (batch, n, dim_feat)
        return torch.sum(x, dim=1)
