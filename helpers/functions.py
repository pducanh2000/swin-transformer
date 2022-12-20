import collections.abc
import itertools
import torch
from torch.utils.checkpoint import checkpoint
from itertools import chain


def window_partition(x: torch.Tensor, window_size: int):
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
    # print(relative_coords)

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


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    From:
    https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/drop.py#L157
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        # >>> model = nn.Sequential(...)
        # >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


if __name__ == "__main__":
    random_tensor = torch.randn((32, 256, 256, 3))
    print(get_relative_position_index(3, 3))
