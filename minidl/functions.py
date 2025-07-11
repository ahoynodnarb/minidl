from __future__ import annotations

import collections.abc as abc
import math
from typing import Callable, Optional, Tuple, Union

import minidiff as md
import minidiff.ops as ops
import minidiff.typing as mdt


def log_sum_exp(x: md.Tensor) -> md.Tensor:
    mx = md.max(x, axis=-1, keepdims=True)
    e = md.exp(x - mx)
    s = md.sum(e, axis=-1, keepdims=True)
    # log-sum-exp take the log of the sum of the exponents shifted by the max, and then shift again later
    lse = mx + md.log(s)
    return lse


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


@ops.unary_op_func(
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


@ops.unary_op_func(
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

    # return view of padded matrix cropped around the padded boundaries
    return mat[:, pad_top : height - pad_bottom, pad_left : width - pad_right, :]


def calculate_same_padding(
    height: int, width: int, kernel_height: int, kernel_width: int, stride: int
) -> Tuple[int, int, int, int]:
    pad_vert = (height * (stride - 1) + kernel_height - stride) / 2
    pad_hori = (width * (stride - 1) + kernel_width - stride) / 2

    # we can evenly pad vertically
    if isinstance(pad_vert, int):
        pad_top = pad_bottom = pad_vert
    else:
        pad_vert = int(math.floor(pad_vert))
        pad_top, pad_bottom = pad_vert, pad_vert + 1

    # we can evenly pad horizontally
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


# layer functions
class Convolve2D(ops.BinaryOpClass):
    def setup(
        self,
        conv_input: md.Tensor,
        kernels: md.Tensor,
        padding: Union[int, float, Tuple[int, int, int, int]] = 0,
        stride: int = 1,
        forward_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        backward_input_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        backward_kern_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
    ):
        _, in_height, in_width, self.in_channels = conv_input.shape
        self.n_kernels, self.kernel_height, self.kernel_width, _ = kernels.shape
        padding = get_padded_edges(padding)
        pad_top, pad_bottom, pad_left, pad_right = padding

        if (in_height - self.kernel_height + pad_top + pad_bottom) % stride != 0:
            raise ValueError("Cannot evenly convolve")
        if (in_height - self.kernel_width + pad_left + pad_right) % stride != 0:
            raise ValueError("Cannot evenly convolve")

        self.padding = padding
        self.stride = stride
        # we need to keep track of the shape of the inputs and outputs so we do not
        # have to recalculate them for every single batch
        self.in_dims = (in_height, in_width)
        self.out_dims = calculate_convolved_dimensions(
            in_height,
            in_width,
            self.kernel_height,
            self.kernel_width,
            self.stride,
            padding=self.padding,
        )
        # we optimize the actual convolution as a large matrix multiplication
        # and we keep track of how the matrices need to be rearranged for that
        # matrix multiplication, also so we don't have to recompute it for each batch
        if forward_indices is None:
            forward_indices = calculate_im2col_indices(
                *self.out_dims, self.kernel_height, self.kernel_width, self.stride
            )
        if backward_input_indices is None:
            backward_input_indices = calculate_im2col_indices(
                *self.in_dims, self.kernel_height, self.kernel_width, self.stride
            )
        if backward_kern_indices is None:
            backward_kern_indices = calculate_im2col_indices(
                self.kernel_height, self.kernel_width, *self.out_dims, self.stride
            )
        self.forward_indices = forward_indices
        self.backward_input_indices = backward_input_indices
        self.backward_kern_indices = backward_kern_indices

    @staticmethod
    def perform_convolution(
        mat: md.Tensor,
        kernels: md.Tensor,
        padding: Optional[Union[int, float, Tuple[int, int, int, int]]] = None,
        stride: int = 1,
        im2col_indices: Optional[Tuple[md.Tensor[int], md.Tensor[int]]] = None,
        out_dims: Optional[Tuple[int, int]] = None,
    ) -> md.Tensor:
        orig_shape = mat.shape
        batch_size, orig_height, orig_width, _ = orig_shape
        n_kernels, kernel_height, kernel_width, kernel_channels = kernels.shape
        if out_dims is None:
            out_dims = calculate_convolved_dimensions(
                orig_height,
                orig_width,
                kernel_height,
                kernel_width,
                stride,
                padding=padding,
            )

        # out_dims is the "physical" dimension of the out matrix,
        # out_shape is the total shape which includes batch size and output channels
        out_shape = (batch_size, *out_dims, n_kernels)

        if padding is not None:
            mat = add_padding(mat, padding=padding)

        # if we're not given the instructions on how to rearrange the matrix,
        # we can fall back to computing it manually
        if im2col_indices is None:
            # calculating the new positions for every element in the input image
            row_indices, col_indices = calculate_im2col_indices(
                *out_dims, kernel_height, kernel_width, stride
            )
        else:
            row_indices, col_indices = im2col_indices

        # filter the input image by these new positions
        as_cols = mat[:, row_indices, col_indices, :]

        # flatten our matrix of kernels
        flattened_kernels = kernels.reshape(
            (n_kernels, kernel_height * kernel_width, kernel_channels)
        )

        # this is the actual convolution step, which is just a single matrix multiplication now!
        convolved = md.tensordot(as_cols, flattened_kernels, axes=((1, 3), (1, 2)))
        reshaped = convolved.reshape(out_shape)
        return reshaped

    def create_forward(self) -> mdt.BinaryFunc:
        def forward(
            conv_input: md.Tensor,
            kernels: md.Tensor,
            padding: Union[int, float, Tuple[int, int, int, int]] = 0,
            stride: int = 1,
            # in_dims: Optional[Tuple[int, ...]] = None,
            # out_dims: Optional[Tuple[int, ...]] = None,
            forward_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
            backward_input_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
            backward_kern_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        ) -> md.Tensor:
            self.setup(
                conv_input,
                kernels,
                padding=padding,
                stride=stride,
                forward_indices=forward_indices,
                backward_input_indices=backward_input_indices,
                backward_kern_indices=backward_kern_indices,
            )
            convolved = Convolve2D.perform_convolution(
                conv_input,
                kernels,
                padding=self.padding,
                stride=self.stride,
                out_dims=self.out_dims,
                im2col_indices=self.forward_indices,
            )
            return convolved

        return forward

    def create_grads(self) -> Tuple[mdt.BinaryOpGrad, mdt.BinaryOpGrad]:
        def compute_grad_wrt_x(
            conv_input: md.Tensor,
            kernels: md.Tensor,
            grad: md.Tensor,
            # padding: Union[int, float, Tuple[int, int, int, int]] = 0,
            # stride: int = 1,
            # in_dims: Optional[Tuple[int, ...]] = None,
            # out_dims: Optional[Tuple[int, ...]] = None,
            # forward_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
            # backward_input_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
            # backward_kern_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        ) -> md.Tensor:
            # _, in_height, in_width, _ = self.conv_input.shape
            # rotate kernels, then swap axes to match up correctly
            # kernel_height, kernel_width = kernels.shape[1], kernels.shape[2]
            # in_dims = (conv_input.shape[1], conv_input.shape[2])
            flipped_kernels = md.flip(md.flip(kernels, axis=1), axis=2)
            flipped_kernels = md.swapaxes(flipped_kernels, -1, 0)

            full_padding = calculate_full_padding(
                kernel_height=self.kernel_height,
                kernel_width=self.kernel_width,
                original_padding=self.padding,
            )
            grad_wrt_x = Convolve2D.perform_convolution(
                grad,
                flipped_kernels,
                padding=full_padding,
                stride=1,
                out_dims=self.in_dims,
                im2col_indices=self.backward_input_indices,
            )

            return grad_wrt_x

        # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
        # the gradient with respect to the weights (kernel) tells us how the loss function changes relative to
        # changes to each individual element of the kernel
        # the overall computation boils down to convolving each channel of the previous outputs by each channel of the gradient
        def compute_grad_wrt_w(
            conv_input: md.Tensor,
            kernels: md.Tensor,
            grad: md.Tensor,
            # padding: Union[int, float, Tuple[int, int, int, int]] = 0,
            # stride: int = 1,
            # in_dims: Optional[Tuple[int, ...]] = None,
            # out_dims: Optional[Tuple[int, ...]] = None,
            # forward_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
            # backward_input_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
            # backward_kern_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        ) -> md.Tensor:
            # normally, computing grad_wrt_w requires you to do convolutions for each slice of the previous outputs
            # and each slice of the gradient. But we can take advantage of batching to instead treat each slice of
            # output as a separate entry to the batch, and each slice of the gradient as a separate "kernel"
            # this results in us having the same final convolution, just the slices end up as the channels instead
            # kernel_height, kernel_width = kernels.shape[1], kernels.shape[2]
            swapped_prev_outputs = md.swapaxes(conv_input, 0, -1)
            swapped_grad = md.swapaxes(grad, 0, -1)
            convolved = Convolve2D.perform_convolution(
                swapped_prev_outputs,
                swapped_grad,
                padding=self.padding,
                stride=self.stride,
                out_dims=(self.kernel_height, self.kernel_width),
                im2col_indices=self.backward_kern_indices,
            )
            grad_wrt_w = md.swapaxes(convolved, 0, -1)

            return grad_wrt_w

        return (compute_grad_wrt_x, compute_grad_wrt_w)


class Dropout(ops.BinaryOpClass):
    def create_forward(self) -> mdt.BinaryFunc:
        def forward(
            inputs: md.Tensor,
            prob: float,
            auto_scale: bool = True,
            trainable: bool = False,
        ) -> md.Tensor:
            if not trainable:
                return inputs
            self.mask = md.binomial(1, 1 - prob, inputs.shape)
            if auto_scale:
                return md.where(self.mask == 0, 0, inputs / prob)
            return md.where(self.mask == 0, 0, inputs)

        return forward

    def create_grads(self) -> Tuple[mdt.BinaryOpGrad, None]:
        def grad_wrt_x(
            inputs: md.Tensor,
            prob: float,
            grad: md.Tensor,
            auto_scale: bool = True,
            trainable: bool = False,
        ) -> md.Tensor:
            if not trainable:
                return grad
            if auto_scale:
                return md.where(self.mask == 0, 0, grad / prob)
            return md.where(self.mask == 0, 0, grad)

        return (grad_wrt_x, None)


class BatchNormalization(ops.TernaryOpClass):
    def create_forward(self) -> mdt.TernaryFunc:
        def forward(
            inputs: md.Tensor,
            gamma: md.Tensor,
            beta: md.Tensor,
            epsilon: float = 1e-3,
            momentum: float = 0.99,
            trainable: bool = False,
            moving_means: Optional[md.Tensor] = None,
            moving_variances: Optional[md.Tensor] = None,
        ) -> md.Tensor:
            n_dimensions = inputs.shape[-1]

            if moving_means is None:
                moving_means = md.zeros(n_dimensions)
            if moving_variances is None:
                moving_variances = md.ones(n_dimensions)

            normalized_dimensions = tuple(range(inputs.ndim - 1))
            dummy_dims = [1] * (len(inputs.shape) - 1)
            gamma_reshaped = gamma.reshape((*dummy_dims, n_dimensions))
            beta_reshaped = beta.reshape((*dummy_dims, n_dimensions))

            if not trainable:
                means_reshaped = moving_means.reshape((*dummy_dims, n_dimensions))
                variances_reshaped = moving_variances.reshape(
                    (*dummy_dims, n_dimensions)
                )

                normalized = (inputs - means_reshaped) / md.sqrt(
                    variances_reshaped + epsilon
                )

                return normalized * gamma_reshaped + beta_reshaped

            means = md.mean(
                inputs, axis=normalized_dimensions, keepdims=True
            )  # returns mu for each dimension of input
            self.mean_deviation = inputs - means
            variances = md.mean(
                md.square(self.mean_deviation),
                axis=normalized_dimensions,
                keepdims=True,
            )  # returns sigma^2 for each input
            self.std_deviation = md.sqrt(variances + epsilon)

            self.x_hat = self.mean_deviation / self.std_deviation

            means_flat = means.reshape(-1)
            variances_flat = variances.reshape(-1)

            moving_means *= momentum
            moving_means += means_flat * (1 - momentum)
            moving_variances *= momentum
            moving_variances += variances_flat * (1 - momentum)

            return gamma_reshaped * self.x_hat + beta_reshaped

        return forward

    def create_grads(
        self,
    ) -> Tuple[mdt.TernaryOpGrad, mdt.TernaryOpGrad, mdt.TernaryOpGrad]:
        def compute_grad_wrt_x(
            inputs: md.Tensor,
            gamma: md.Tensor,
            beta: md.Tensor,
            grad: md.Tensor,
            epsilon: float = 1e-3,
            momentum: float = 0.99,
            trainable: bool = False,
            moving_means: Optional[md.Tensor] = None,
            moving_variances: Optional[md.Tensor] = None,
        ) -> md.Tensor:
            if not trainable:
                n_dimensions = inputs.shape[-1]
                if moving_variances is None:
                    moving_variances = md.ones(n_dimensions)
                dummy_dims = [1] * (len(inputs.shape) - 1)
                gamma_reshaped = gamma.reshape((*dummy_dims, n_dimensions))
                variances_reshaped = moving_variances.reshape(
                    (*dummy_dims, n_dimensions)
                )

                normalized = 1 / md.sqrt(variances_reshaped + epsilon)

                return grad * normalized * gamma_reshaped

            ndims = len(grad.shape)
            norm_axes = tuple(range(ndims - 1))  # (0, 1, 2) for images
            m = math.prod([grad.shape[axis] for axis in norm_axes])

            gamma_reshaped = gamma.reshape((1,) * (ndims - 1) + (-1,))

            # dL/dx_hat
            grad_x_hat = grad * gamma_reshaped

            # dL/dvar
            grad_var = md.sum(
                grad_x_hat * self.mean_deviation * -0.5 * (self.std_deviation**-3),
                axis=norm_axes,
                keepdims=True,
            )

            # dL/dmean
            term1 = md.sum(
                -grad_x_hat / self.std_deviation, axis=norm_axes, keepdims=True
            )
            term2 = (
                grad_var
                * md.sum(-2 * self.mean_deviation, axis=norm_axes, keepdims=True)
                / m
            )
            grad_mean = term1 + term2

            # dL/dx components
            component1 = grad_x_hat / self.std_deviation
            component2 = grad_var * 2 * self.mean_deviation / m
            component3 = grad_mean / m

            grad_input = component1 + component2 + component3

            return grad_input

        def compute_grad_wrt_gamma(
            inputs: md.Tensor,
            gamma: md.Tensor,
            beta: md.Tensor,
            grad: md.Tensor,
            epsilon: float = 1e-3,
            momentum: float = 0.99,
            trainable: bool = False,
            moving_means: Optional[md.Tensor] = None,
            moving_variances: Optional[md.Tensor] = None,
        ) -> md.Tensor:
            if not trainable:
                n_dimensions = inputs.shape[-1]
                if moving_variances is None:
                    moving_variances = md.ones(n_dimensions)
                if moving_means is None:
                    moving_means = md.zeros(n_dimensions)

                dummy_dims = [1] * (len(inputs.shape) - 1)
                means_reshaped = moving_means.reshape((*dummy_dims, n_dimensions))
                variances_reshaped = moving_variances.reshape(
                    (*dummy_dims, n_dimensions)
                )

                normalized = (inputs - means_reshaped) / md.sqrt(
                    variances_reshaped + epsilon
                )
                return grad * normalized

            normalized_dimensions = tuple(range(grad.ndim - 1))
            return md.sum(grad * self.x_hat, axis=normalized_dimensions)

        def compute_grad_wrt_beta(
            inputs: md.Tensor,
            gamma: md.Tensor,
            beta: md.Tensor,
            grad: md.Tensor,
            epsilon: float = 1e-3,
            momentum: float = 0.99,
            trainable: bool = False,
            moving_means: Optional[md.Tensor] = None,
            moving_variances: Optional[md.Tensor] = None,
        ) -> md.Tensor:

            normalized_dimensions = tuple(range(grad.ndim - 1))
            if not trainable:
                return md.sum(grad, axis=normalized_dimensions, keepdims=True)

            return md.sum(grad, axis=normalized_dimensions)

        return (compute_grad_wrt_x, compute_grad_wrt_gamma, compute_grad_wrt_beta)


class MaxPooling2D(ops.BinaryOpClass):
    def setup(self, inputs: md.Tensor, pool_size: int, stride: Optional[int] = None):
        # self.inputs = inputs
        # self.pool_size = pool_size
        self.stride = pool_size if stride is None else stride

        _, in_height, in_width, _ = inputs.shape
        self.in_dims = (in_height, in_width)
        self.out_dims = calculate_convolved_dimensions(
            in_height, in_width, pool_size, pool_size, self.stride
        )

        # self.in_channels = in_channels

        self.forward_indices = calculate_im2col_indices(
            *self.out_dims, pool_size, pool_size, self.stride
        )

        self.row_offset, self.col_offset = self.compute_window_offsets(
            *self.out_dims, self.stride
        )

    # this entire function essentially just computes the top left corner of each patch
    def compute_window_offsets(
        self, out_height: int, out_width: int, stride: int
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

    def create_forward(self) -> mdt.BinaryFunc:
        def forward(
            inputs: md.Tensor,
            pool_size: int,
            stride: Optional[int] = None,
        ) -> md.Tensor:
            self.setup(inputs, pool_size, stride=stride)

            batch_size, _, _, in_channels = inputs.shape

            row_indices, col_indices = self.forward_indices

            # transform inputs into column vectors of patches
            as_cols = inputs[:, row_indices, col_indices, :]

            flat_indices = md.argmax(as_cols, axis=1, keepdims=True)

            # row_offset, col_offset = self.compute_window_offsets(
            #     *self.out_dims, self.stride
            # )
            # add precomputed offsets to the indices since flat_indices gives coordinates relative to individual patches.
            # but we need indices relative to the entire input matrix
            row_max_indices = flat_indices // pool_size + self.row_offset
            col_max_indices = flat_indices % pool_size + self.col_offset

            # need these indices as placeholders to correctly index
            batch_indices = md.arange(batch_size)[
                ..., md.newaxis, md.newaxis, md.newaxis
            ]  # shape: (batch_size, 1, 1, 1)
            channel_indices = md.arange(in_channels)[
                md.newaxis, md.newaxis, md.newaxis, ...
            ]  # shape: (1, 1, 1, in_channels)

            out_shape = (batch_size, *self.out_dims, in_channels)
            # finally, actually index and return this
            max_values = inputs[
                batch_indices, row_max_indices, col_max_indices, channel_indices
            ].reshape(out_shape)
            self.prev_indices = (
                batch_indices,
                row_max_indices,
                col_max_indices,
                channel_indices,
            )

            return max_values

        return forward

    def create_grads(self) -> Tuple[mdt.BinaryOpGrad, None]:
        def compute_grad_wrt_x(
            inputs: md.Tensor,
            pool_size: int,
            grad: md.Tensor,
            # stride: Optional[int] = None,
            # out_dims: Optional[Tuple[int, ...]] = None,
            # forward_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        ) -> md.Tensor:
            batch_size, _, _, in_channels = inputs.shape

            batch_indices, row_max_indices, col_max_indices, channel_indices = (
                self.prev_indices
            )

            # the gradient for every element that is not the max is zeroed out, this is kind of
            # like a blank canvas before we paint on the gradients
            zeros = md.zeros_like(inputs)
            flattened_grad = grad.reshape((batch_size, 1, -1, in_channels))

            # similar to forward function, just assigning at these indices instead
            # this is the "painting" step
            zeros[batch_indices, row_max_indices, col_max_indices, channel_indices] = (
                flattened_grad
            )

            return zeros

        return (compute_grad_wrt_x, None)


class MeanPooling2D(ops.BinaryOpClass):
    def setup(
        self,
        inputs: md.Tensor,
        pool_size: int,
        stride: Optional[int] = None,
    ):
        # self.inputs = inputs
        _, in_height, in_width, _ = inputs.shape
        self.stride = pool_size if stride is None else stride

        self.in_dims = (in_height, in_width)
        self.out_dims = calculate_convolved_dimensions(
            in_height, in_width, pool_size, pool_size, self.stride
        )

        self.forward_indices = calculate_im2col_indices(
            *self.out_dims, pool_size, pool_size, self.stride
        )

    def create_forward(self) -> mdt.BinaryFunc:
        def forward(
            inputs: md.Tensor,
            pool_size: int,
            stride: Optional[int] = None,
        ) -> md.Tensor:
            self.setup(inputs, pool_size, stride=stride)
            batch_size, _, _, in_channels = inputs.shape

            out_shape = (batch_size, *self.out_dims, in_channels)

            row_indices, col_indices = self.forward_indices

            # transform inputs into column vectors of patches
            as_cols = inputs[:, row_indices, col_indices, :]

            averaged = md.mean(as_cols, axis=1, keepdims=True)
            return averaged.reshape(out_shape)

        return forward

    def create_grads(self) -> Tuple[mdt.BinaryOpGrad, None]:
        def compute_grad_wrt_x(
            inputs: md.Tensor,
            pool_size: int,
            grad: md.Tensor,
        ) -> md.Tensor:
            batch_size, grad_height, grad_width, in_channels = grad.shape
            # add the extra 1 so that indexing broadcasts for the entire patch
            flattened_grad = grad.reshape(
                (batch_size, 1, grad_height * grad_width, in_channels)
            )
            grad_wrt_x = md.zeros_like(inputs)
            row_indices, col_indices = self.forward_indices
            grad_wrt_x[:, row_indices, col_indices, :] = flattened_grad / (
                pool_size * pool_size
            )
            return grad_wrt_x

        return (compute_grad_wrt_x, None)


# loss functions
class CrossEntropy(ops.BinaryOpClass):
    def setup(
        self,
        y_true: md.Tensor,
        y_pred: md.Tensor,
        from_logits: bool = False,
        smoothing: Union[int, float] = 0,
    ):
        if y_true is None:
            raise ValueError("Empty ground truth array")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        if smoothing == 0:
            self.y_true = y_true
        else:
            n_classes = y_true.shape[-1]
            self.y_true = (1 - smoothing) * y_true + (smoothing / n_classes)

        if from_logits:
            self.y_pred = y_pred
        else:
            # using more unstable method, need to avoid division by 0
            self.y_pred = y_pred.clip(1e-8, None)

        self.from_logits = from_logits

    def create_forward(self) -> mdt.BinaryFunc:
        # formula for cross entropy loss is sum(y_true * -log(y_pred))
        def forward(
            y_true: md.Tensor,
            y_pred: md.Tensor,
            from_logits: bool = False,
            smoothing: Union[int, float] = 0,
        ) -> md.Tensor:
            self.setup(y_true, y_pred, from_logits=from_logits, smoothing=smoothing)
            if from_logits:
                lse = log_sum_exp(self.y_pred)
                loss = -(self.y_true * (self.y_pred - lse))
            else:
                loss = -(self.y_true * md.log(self.y_pred))

            return md.sum(loss, axis=-1, keepdims=True)

        return forward

    def create_grads(self) -> Tuple[None, mdt.BinaryOpGrad]:
        def compute_grad_wrt_x(y_true, y_pred, grad) -> md.Tensor:
            # more numerically stable than -y_true / y_pred
            # don't need to sum these since they'll be automatically broadcasted in the backward pass
            if self.from_logits:
                mx = md.max(self.y_pred, axis=-1, keepdims=True)
                e = md.exp(self.y_pred - mx)
                softmax = e / md.sum(e, axis=-1, keepdims=True)
                loss_grad = grad * (softmax - self.y_true)
            else:
                loss_grad = grad * -self.y_true / self.y_pred

            return loss_grad

        return (None, compute_grad_wrt_x)


class BinaryCrossEntropy(ops.BinaryOpClass):
    def setup(
        self,
        y_true: md.Tensor,
        y_pred: md.Tensor,
        from_logits: bool = False,
        smoothing: Union[int, float] = 0,
    ):
        if y_true is None:
            raise ValueError("Empty ground truth array")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        if smoothing <= 0:
            self.y_true = y_true
        else:
            n_classes = y_true.shape[-1]
            self.y_true = (1 - smoothing) * y_true + (smoothing / n_classes)

        if from_logits:
            self.y_pred = y_pred
        else:
            # using more unstable method, need to avoid division by 0
            self.y_pred = y_pred.clip(1e-8, None)

        self.from_logits = from_logits

    def create_forward(self) -> mdt.BinaryFunc:
        def forward(
            y_true: md.Tensor,
            y_pred: md.Tensor,
            from_logits: bool = False,
            smoothing: Union[int, float] = 0,
        ) -> md.Tensor:
            self.setup(y_true, y_pred, from_logits=from_logits, smoothing=smoothing)
            if self.from_logits:
                loss = (
                    md.log(1 + md.exp(-self.y_pred)) + (1 - self.y_true) * self.y_pred
                )
            else:
                loss = -(
                    self.y_true * md.log(self.y_pred)
                    + (1 - self.y_true) * md.log(1 - self.y_pred)
                )

            return md.mean(loss, axis=-1, keepdims=True)

        return forward

    def create_grads(self) -> Tuple[None, mdt.BinaryOpGrad]:
        def compute_grad_wrt_x(grad: md.Tensor) -> md.Tensor:
            if self.from_logits:
                loss_grad = grad * -(self.y_true - self.y_pred) / self.y_true.shape[-1]
            else:
                loss_grad = (
                    grad
                    * -((self.y_true - self.y_pred) / (self.y_pred * (1 - self.y_pred)))
                    / self.y_true.shape[-1]
                )

            return loss_grad

        return (None, compute_grad_wrt_x)


class MeanSquaredError(ops.BinaryOpClass):
    def create_forward(self) -> mdt.BinaryFunc:
        def forward(y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
            return md.mean((y_true - y_pred) ** 2, axis=-1)

        return forward

    def create_grads(self) -> Tuple[None, mdt.BinaryOpGrad]:
        def compute_grad_wrt_x(
            y_true: md.Tensor, y_pred: md.Tensor, grad: md.Tensor
        ) -> md.Tensor:
            return grad * 2 * (y_pred - y_true) / y_true.shape[-1]

        return (None, compute_grad_wrt_x)


convolve2d: Callable[[md.Tensor, md.Tensor], md.Tensor] = ops.create_stateful_op_func(
    op_class=Convolve2D, tensor_only=True, op_name="convolve2d"
)
dropout: Callable[[md.Tensor, float], md.Tensor] = ops.create_stateful_op_func(
    op_class=Dropout,
    op_name="dropout",
)
batchnormalize: Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor] = (
    ops.create_stateful_op_func(
        op_class=BatchNormalization,
        propagate_kwargs=True,
        tensor_only=True,
        op_name="batchnormalize",
    )
)
maxpool2d: Callable[[md.Tensor, int], md.Tensor] = ops.create_stateful_op_func(
    op_class=MaxPooling2D,
    op_name="maxpool2d",
)
meanpool2d: Callable[[md.Tensor, int], md.Tensor] = ops.create_stateful_op_func(
    op_class=MeanPooling2D,
    op_name="meanpool2d",
)
cross_entropy: Callable[[md.Tensor, md.Tensor], md.Tensor] = (
    ops.create_stateful_op_func(
        op_class=CrossEntropy,
        tensor_only=True,
        op_name="cross_entropy",
    )
)
binary_cross_entropy: Callable[[md.Tensor, md.Tensor], md.Tensor] = (
    ops.create_stateful_op_func(
        op_class=BinaryCrossEntropy,
        tensor_only=True,
        op_name="binary_cross_entropy",
    )
)
mean_squared_error: Callable[[md.Tensor, md.Tensor], md.Tensor] = (
    ops.create_stateful_op_func(
        op_class=MeanSquaredError,
        tensor_only=True,
        op_name="mean_squared_error",
    )
)

__all__ = [
    "convolve2d",
    "dropout",
    "batchnormalize",
    "maxpool2d",
    "meanpool2d",
    "cross_entropy",
    "binary_cross_entropy",
    "mean_squared_error",
]
