import torch
import torch.nn as nn

from helpers.functions import to_ntuple, get_relative_position_index
from helpers.weight_init import trunc_normal_


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        """ Window based multi-head self attention (W-MSA) module with relative position bias.
            It supports both of shifted and non-shifted window.
            Args:
                dim (int): Number of input channels.
                num_heads (int): Number of attention heads.
                head_dim (int): Number of channels per head (dim // num_heads if not set)
                window_size (tuple[int]): The height and width of the window.
                qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
                attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
                proj_drop (float, optional): Dropout ratio of output. Default: 0.0
            """

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        attn_dim = num_heads * head_dim

        to_2tuple = to_ntuple(2)
        self.window_size = to_2tuple(window_size)  # (window_h, window_w)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.scale = 1 / torch.sqrt(self.head_dim)

        # define a parameter table of relative position bias shape: (2*window_h-1) * (2*window_w-1), num_heads
        self.relative_position_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), self.num_heads))

        # get pairwise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_table[self.relative_position_index.view(-1)].view(
            self.window_area,
            self.window_area,
            -1
        )  # (win_h * win_w, win_h * win_w, num_heads)
        relative_position_bias = relative_position_bias.permute(1, 0, 2)  # (num_heads, window_area, window_area)
        return relative_position_bias.contiguous().unsqueeze(0)  # (1, num_heads, window_area, window_area)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # C = self.dim
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1)     # (B_, N, 3, num_heads, attn_dim//num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, attn_dim//num_heads)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # attn_score  = softmax(Q.Kt) / sqrt(d_k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)      # (B_, num_heads, N, N)
        attn = attn + self._get_rel_pos_bias()

        # In case masked multi-head attention
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)      # (B, num_heads, N, N)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # attn_value = attn_score @ v
        attn = attn @ v   # (B_, num_heads, N, head_dim)
        attn = attn.transpose(1, 2).reshape(B_, N, -1)      # (B_, N, num_head * head_dims=attn_dim)

        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn


class SwinTransformerBlock(nn.Module):
    def __init__(self, ):
