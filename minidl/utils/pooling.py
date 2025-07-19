from __future__ import annotations

import collections.abc as abc
import math
from typing import TYPE_CHECKING

import minidiff as md

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union


# formula for a convolved dimension is just (dim - kernel_dim + 2 * padding) / stride + 1
def calculate_convolved_dimensions(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    stride: int,
    padding: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int]:
    if padding is None:
        top = bottom = left = right = 0
    else:
        top, bottom, left, right = padding
    vertical_padding = top + bottom
    horizontal_padding = left + right

    out_height = (height - kernel_height + vertical_padding) // stride + 1
    out_width = (width - kernel_width + horizontal_padding) // stride + 1
    return (int(out_height), int(out_width))


# this returns indices that we use to index over a matrix of rows_out x cols_out so that it is im2col'ed
def calculate_im2col_indices(
    rows_out: int, cols_out: int, kernel_height: int, kernel_width: int, stride: int
) -> Tuple[md.Tensor, md.Tensor]:
    # these are the indices that correspond to each row within the patch
    kernel_row_indices = md.repeat(md.arange(kernel_height), kernel_width)
    # these are the indices corresponding to the row portion of the position of each patch within the input matrix
    conv_row_indices = stride * md.repeat(md.arange(rows_out), cols_out)

    # these are the indices that correspond to each column within the patch
    kernel_col_indices = md.tile(md.arange(kernel_width), kernel_height)
    # these are the indices that correspond to the column portion of the position of each patch within the input matrix
    conv_col_indices = stride * md.tile(md.arange(cols_out), rows_out)

    row_indices = kernel_row_indices.reshape((-1, 1)) + conv_row_indices.reshape(
        (1, -1)
    )
    col_indices = kernel_col_indices.reshape((-1, 1)) + conv_col_indices.reshape(
        (1, -1)
    )

    return (row_indices, col_indices)


def get_padded_edges(
    padding: Union[int, float, Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    # padding is already a tuple
    if isinstance(padding, abc.Sequence):
        return padding
    # padding is an integer
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    # padding is a float
    padding = int(math.floor(padding))
    pad_top = pad_left = padding
    pad_bottom = pad_right = padding + 1

    return pad_top, pad_bottom, pad_left, pad_right


@md.ops.unary_op_func(
    grad=lambda a, grad, padding=None: remove_padding(grad, padding=padding),
    propagate_kwargs=True,
)
def add_padding(
    mat: md.Tensor,
    padding: Optional[Tuple[int, int, int, int]] = None,
) -> md.Tensor:
    batch_size, height, width, channels = mat.shape

    if (
        padding is None
        or (isinstance(padding, abc.Sequence) and sum(padding) == 0)
        or padding == 0
    ):
        return mat

    pad_top, pad_bottom, pad_left, pad_right = padding

    padded = md.zeros(
        (
            batch_size,
            pad_top + height + pad_bottom,
            pad_left + width + pad_right,
            channels,
        )
    )

    padded[:, pad_top : height + pad_top, pad_left : width + pad_left, :] = mat
    return padded


@md.ops.unary_op_func(
    grad=lambda a, grad, padding=None: add_padding(grad, padding=padding),
    propagate_kwargs=True,
)
def remove_padding(
    mat: md.Tensor,
    padding: Optional[Tuple[int, int, int, int]] = None,
) -> md.Tensor:
    _, height, width, _ = mat.shape

    if (
        padding is None
        or (isinstance(padding, abc.Sequence) and sum(padding) == 0)
        or padding == 0
    ):
        return mat

    pad_top, pad_bottom, pad_left, pad_right = padding

    return mat[:, pad_top : height - pad_bottom, pad_left : width - pad_right, :]


def calculate_same_padding(
    height: int, width: int, kernel_height: int, kernel_width: int, stride: int
) -> Tuple[int, int, int, int]:
    pad_vert = (height * (stride - 1) + kernel_height - stride) / 2
    pad_hori = (width * (stride - 1) + kernel_width - stride) / 2

    if isinstance(pad_vert, int):
        pad_top = pad_bottom = pad_vert
    else:
        pad_vert = int(math.floor(pad_vert))
        pad_top, pad_bottom = pad_vert, pad_vert + 1

    if isinstance(pad_hori, int):
        pad_left = pad_right = pad_hori
    else:
        pad_hori = int(math.floor(pad_hori))
        pad_left, pad_right = pad_hori, pad_hori + 1

    return (pad_top, pad_bottom, pad_left, pad_right)


# formula for full padding is just kernel_dim - original_pad_dim - 1
def calculate_full_padding(
    kernel_height: int,
    kernel_width: int,
    original_padding: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    pad_top = kernel_height - 1
    pad_bottom = kernel_height - 1
    pad_left = kernel_width - 1
    pad_right = kernel_width - 1

    o_top, o_bottom, o_left, o_right = original_padding
    pad_top -= o_top
    pad_bottom -= o_bottom
    pad_left -= o_left
    pad_right -= o_right
    return (pad_top, pad_bottom, pad_left, pad_right)
