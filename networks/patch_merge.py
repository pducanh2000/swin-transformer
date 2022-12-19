import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else 4 * self.dim
        self.reduction = nn.Linear(4 * self.dim, self.out_dim)
        self.norm = norm_layer(4 * self.dim)

    def forward(self, x):
        """
        :param x: the input tensor with the shape of (B, H*W, C)
        :return:
        """

        H, W = self.input_resolution
        B, L, C = x.size()

        assert L == H*W, "Input has wrong size"
        assert (H % 2 == 0 and W % 2 == 0), f"x size ({H}, {W}) are not even"

        x = x.view(B, H, W, C)          # (B ,H, W, C)
        x1 = x[:, 0::2, 0::2, :]        # (B, H//2, W//2, C)
        x2 = x[:, 1::2, 0::2, :]        # (B, H//2, W//2, C)
        x3 = x[:, 0::2, 1::2, :]        # (B, H//2, W//2, C)
        x4 = x[:, 1::2, 1::2, :]        # (B, H//2, W//2, C)

        x = torch.stack([x1, x2, x3, x4], dim=-1)   # (B, H//2, W//2, 4 * C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)        # (B, H//2 * W//2, 4 * C)
        x = self.reduction(x)   # (B, H//2 * W//2, self.out_dim)

        return x
