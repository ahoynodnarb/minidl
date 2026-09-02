from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union

import minidiff as md

from minidl.utils.pooling import (
    add_padding,
    calculate_convolved_dimensions,
    calculate_im2col_indices,
    get_padded_edges,
)


def _log_sum_exp(x: md.Tensor) -> md.Tensor:
    mx = md.max(x, axis=-1, keepdims=True)
    e = md.exp(x - mx)
    s = md.sum(e, axis=-1, keepdims=True)
    # log-sum-exp take the log of the sum of the exponents shifted by the max, and then shift again later
    lse = mx + md.log(s)
    return lse


def _preprocess_inputs(
    y_true: md.Tensor, y_pred: md.Tensor, from_logits: bool, smoothing: float
) -> Tuple[md.Tensor, md.Tensor]:
    if smoothing > 0:
        n_classes = y_true.shape[-1]
        y_true = (1 - smoothing) * y_true + (smoothing / n_classes)

    if not from_logits:
        # using more unstable method, need to avoid division by 0
        y_pred = y_pred.clip(1e-8, None)

    return y_true, y_pred


# layer functions


def convolve2d(
    x: md.Tensor,
    kernels: md.Tensor,
    padding: Union[int, float, Tuple[int, int, int, int]] = 0,
    stride: int = 1,
    im2col_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
) -> md.Tensor:
    batch_size, in_height, in_width, _ = x.shape
    n_kernels, kernel_height, kernel_width, kernel_channels = kernels.shape

    padding = get_padded_edges(padding)
    pad_top, pad_bottom, pad_left, pad_right = padding

    if (in_height - kernel_height + pad_top + pad_bottom) % stride != 0:
        raise ValueError("Cannot evenly convolve")
    if (in_height - kernel_width + pad_left + pad_right) % stride != 0:
        raise ValueError("Cannot evenly convolve")

    x = add_padding(x, padding=padding)

    out_dims = calculate_convolved_dimensions(
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride,
        padding=padding,
    )
    # we optimize the actual convolution as a large matrix multiplication
    # and we keep track of how the matrices need to be rearranged for that
    # matrix multiplication, also so we don't have to recompute it for each batch
    if im2col_indices is None:
        im2col_indices = calculate_im2col_indices(
            *out_dims, kernel_height, kernel_width, stride
        )

    row_indices, col_indices = im2col_indices

    # out_dims is the "physical" dimension of the out matrix,
    # out_shape is the total shape which includes batch size and output channels
    out_shape = (batch_size, *out_dims, n_kernels)

    # filter the input image by these new positions
    as_cols = x[:, row_indices, col_indices, :]

    # flatten our matrix of kernels
    flattened_kernels = kernels.reshape(
        (n_kernels, kernel_height * kernel_width, kernel_channels)
    )

    # this is the actual convolution step, which is just a single matrix multiplication now!
    convolved = md.tensordot(as_cols, flattened_kernels, axes=((1, 3), (1, 2)))
    reshaped = convolved.reshape(out_shape)
    return reshaped


def dropout(
    x: md.Tensor,
    prob: float,
    auto_scale: bool = True,
    trainable: bool = False,
) -> md.Tensor:
    if not trainable:
        return x

    mask = md.binomial(1, 1 - prob, x.shape)
    if auto_scale:
        return md.where(mask == 0, 0, x / (1 - prob))

    return md.where(mask == 0, 0, x)


def batch_normalize(
    x: md.Tensor,
    gamma: md.Tensor,
    beta: md.Tensor,
    epsilon: float = 1e-7,
    momentum: float = 0.99,
    trainable: bool = False,
    moving_means: Optional[md.Tensor] = None,
    moving_variances: Optional[md.Tensor] = None,
) -> md.Tensor:
    n_dimensions = x.shape[-1]

    if moving_means is None:
        moving_means = md.zeros(n_dimensions)
    if moving_variances is None:
        moving_variances = md.ones(n_dimensions)

    normalized_dimensions = tuple(range(x.ndim - 1))
    dummy_dims = [1] * (len(x.shape) - 1)
    gamma_reshaped = gamma.reshape((*dummy_dims, n_dimensions))
    beta_reshaped = beta.reshape((*dummy_dims, n_dimensions))

    if not trainable:
        means_reshaped = moving_means.reshape((*dummy_dims, n_dimensions))
        variances_reshaped = moving_variances.reshape((*dummy_dims, n_dimensions))

        normalized = (x - means_reshaped) / md.sqrt(variances_reshaped + epsilon)
        return normalized * gamma_reshaped + beta_reshaped

    # --- train mode: use current batch statistics, update the running stats ---
    normalized_dimensions = tuple(range(x.ndim - 1))
    means = md.mean(x, axis=normalized_dimensions, keepdims=True)
    mean_deviation = x - means
    variances = md.mean(mean_deviation**2, axis=normalized_dimensions, keepdims=True)

    std_deviation = md.sqrt(variances + epsilon)
    x_hat = mean_deviation / std_deviation

    means_flat = means.ravel()
    variances_flat = variances.ravel()

    moving_means *= momentum
    moving_means += means_flat * (1 - momentum)
    moving_variances *= momentum
    moving_variances += variances_flat * (1 - momentum)

    return gamma_reshaped * x_hat + beta_reshaped


def _compute_window_offsets(
    out_height: int, out_width: int, stride: int
) -> Tuple[md.Tensor, md.Tensor]:
    # number of pools is just how many can fit in within the cropped area
    n_pools = out_height * out_width

    # this is just 0,1,2,...n_pools - 1. It is just the position of each pool
    pool_indices = md.arange(n_pools)[..., md.newaxis]

    # finally the actual offsets.
    # window_rows represents the row of the top left corner of the pool
    # window_cols represents the column of the top left corner of the pool
    window_rows = (pool_indices // out_width) * stride
    window_cols = (pool_indices % out_width) * stride

    return (window_rows, window_cols)


def max_pool2d(
    x: md.Tensor,
    pool_size: int,
    stride: Optional[int] = None,
    im2col_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
) -> md.Tensor:
    stride = pool_size if stride is None else stride

    batch_size, in_height, in_width, in_channels = x.shape
    out_dims = calculate_convolved_dimensions(
        in_height, in_width, pool_size, pool_size, stride
    )

    if im2col_indices is None:
        im2col_indices = calculate_im2col_indices(
            *out_dims, pool_size, pool_size, stride
        )

    row_indices, col_indices = im2col_indices

    row_offset, col_offset = _compute_window_offsets(*out_dims, stride)

    as_cols = x[:, row_indices, col_indices, :]

    flat_indices = md.argmax(as_cols, axis=1, keepdims=True)
    # add precomputed offsets to the indices since flat_indices gives coordinates relative to individual patches.
    # but we need indices relative to the entire input matrix
    row_max_indices = flat_indices // pool_size + row_offset
    col_max_indices = flat_indices % pool_size + col_offset

    batch_indices = md.arange(batch_size)[
        ..., md.newaxis, md.newaxis, md.newaxis
    ]  # shape: (batch_size, 1, 1, 1)
    channel_indices = md.arange(in_channels)[
        md.newaxis, md.newaxis, md.newaxis, ...
    ]  # shape: (1, 1, 1, in_channels)

    out_shape = (batch_size, *out_dims, in_channels)
    # finally, actually index and return this
    max_values = x[
        batch_indices, row_max_indices, col_max_indices, channel_indices
    ].reshape(out_shape)

    return max_values


# loss functions
def cross_entropy(
    y_true: md.Tensor,
    y_pred: md.Tensor,
    from_logits: bool = False,
    smoothing: float = 0.0,
) -> md.Tensor:
    y_true, y_pred = _preprocess_inputs(
        y_true, y_pred, from_logits=from_logits, smoothing=smoothing
    )

    if from_logits:
        lse = _log_sum_exp(y_pred)
        loss = -(y_true * (y_pred - lse))
    else:
        loss = -(y_true * md.log(y_pred))

    return md.sum(loss, axis=-1, keepdims=True) / len(y_true)


def binary_cross_entropy(
    y_true: md.Tensor,
    y_pred: md.Tensor,
    from_logits: bool = False,
    smoothing: float = 0.0,
) -> md.Tensor:
    y_true, y_pred = _preprocess_inputs(
        y_true, y_pred, from_logits=from_logits, smoothing=smoothing
    )

    if from_logits:
        loss = md.log(1 + md.exp(-y_pred)) + (1 - y_true) * y_pred
    else:
        loss = -(y_true * md.log(y_pred) + (1 - y_true) * md.log(1 - y_pred))

    return md.sum(loss, axis=-1, keepdims=True) / len(y_true)


def mean_squared_error(
    y_true: md.Tensor,
    y_pred: md.Tensor,
) -> md.Tensor:
    return md.sum((y_true - y_pred) ** 2) / len(y_true)


def softmax(x: md.Tensor) -> md.Tensor:
    # subtracting the maximum keeps the exponentiated values low
    # after doing the algebra, the results are the same
    mx = md.max(x, axis=-1, keepdims=True)
    exponentiated = md.exp(x - mx)
    return exponentiated / md.sum(exponentiated, axis=-1, keepdims=True)


def relu(x: md.Tensor) -> md.Tensor:
    return x.clip(0, None)


def leakyrelu(x: md.Tensor, alpha: float = 0.01) -> md.Tensor:
    scaled = alpha * x
    return md.where(x >= 0, x, scaled)


def sigmoid(x: md.Tensor) -> md.Tensor:
    x = x.clip(-500, 500)
    return 1 / (1 + md.exp(-x))


__all__ = [
    "convolve2d",
    "dropout",
    "batchnormalize",
    "maxpool2d",
    "cross_entropy",
    "binary_cross_entropy",
    "mean_squared_error",
    "softmax",
    "relu",
    "leakyrelu",
    "sigmoid",
]
