import torch
import torch.nn as nn
from helpers.functions import to_ntuple


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


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        to_2tuple = to_ntuple(2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
