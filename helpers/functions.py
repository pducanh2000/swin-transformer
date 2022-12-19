import collections.abc
import itertools
import torch


def patch_partition(x: torch.Tensor, window_size: int):
    """
    :param x: tensor with the shape of (B, H, W, C)
    :param window_size: size of the window
    :return: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.size()
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # windows' shape(B * H//window_size * W//window_size, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int):
    """
    :param windows: (B * num_windows, window_size, window_size, C)
    :param window_size: size og the window
    :param height: height of the image feature
    :param width: width of the image feature
    :return: tensor with the shape of (B, H, W, C)
    """
    batch_num_windows = windows.size(0)
    batch_size = int(batch_num_windows / ((height // window_size) * (width // window_size)))
    windows = windows.view(batch_size, height//window_size, width//window_size, window_size, window_size, -1)
    x = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return x


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w), indexing="ij"))    # (2, win_h, win_w)
    # print(coords)
    coords_flatten = torch.flatten(coords, 1)   # (2, win_h * win_w)
    # print(coords_flatten)
    relative_coords = coords_flatten[:, None, :] - coords_flatten[:, :, None]      # (2, win_h * win_w, win_h * win_w)
    # print(relative_coords)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()         # ( win_h * win_w, win_h * win_w, 2)

    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1

    return relative_coords.sum(-1)          # (win_h * win_w, win_h * win_w)


def to_ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(itertools.repeat(x, n))
    return parse


if __name__ == "__main__":
    random_tensor = torch.randn((32, 256, 256, 3))
    print(get_relative_position_index(3, 3))
