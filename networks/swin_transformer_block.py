import torch
import torch.nn as nn

from helpers.functions import to_ntuple, get_relative_position_index, drop_path, window_partition, window_reverse
from helpers.weight_init import trunc_normal_


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        to_2tuple = to_ntuple(2)
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


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
        B_, N, C = x.shape  # C = self.dim, N = window_size * window_size
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1)     # (B_, N, 3, num_heads, attn_dim//num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, attn_dim//num_heads)
        q, k, v = qkv.unbind(0)  # make torch script happy (cannot use tensor as tuple)

        # attn_score  = softmax(Q.Kt) / sqrt(d_k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)      # (B_, num_heads, N, N)
        attn = attn + self._get_rel_pos_bias()

        # In case masked multi-head attention
        if mask is not None:
            # attn_mask is shape of (num_win, window_size * window_size, window_size * window_size)
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)      # (B, num_heads, N, N)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # attn_value = attn_score @ v
        attn = attn @ v   # (B_, num_heads, N, head_dim)
        attn = attn.transpose(1, 2).reshape(B_, N, -1)      # (B_, N, num_head * head_dims=attn_dim)

        attn = self.proj(attn)              # (B_, N, dim)
        attn = self.proj_drop(attn)         # (B_, N, dim)
        return attn


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=4, head_dims=None, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size

        if min(input_resolution) <= self.window_size:
            # if the input_resolution is smaller than the window size, we do not need to partition to windows
            self.shift_size = 0
            self.window_size = min(input_resolution)

        assert 0 <= self.shift_size < self.window_size  # shift size should be in the range of [0, window_size)

        self.norm1 = norm_layer(dim)
        to_2tuple = to_ntuple(2)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, head_dim=head_dims,
                                    window_size=to_2tuple(window_size), qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # attn_mask is shape of (num_win, window_size * window_size, window_size * window_size)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert(L == H * W, "input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


