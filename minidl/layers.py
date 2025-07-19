from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Sequence, Tuple, Union, BinaryIO

    import minidiff.typing as mdt

import minidiff as md

from minidl import functions as F
from minidl.utils.pooling import (
    calculate_convolved_dimensions,
    calculate_im2col_indices,
    get_padded_edges,
)


class Layer:
    trainable: bool = False

    def forward(self, x: md.Tensor) -> md.Tensor:
        raise NotImplementedError

    def __call__(self, x: md.Tensor) -> md.Tensor:
        return self.forward(x)


class OptimizableLayer(Layer):
    def __init__(self, l2_lambda: float = 0.0):
        self.l2_lambda = l2_lambda
        self.params = []

    def save_layer(self, fstream: BinaryIO):
        raise NotImplementedError

    def load_layer(self, fstream: BinaryIO):
        raise NotImplementedError

    def setup(self, trainable: bool = True, reset_params: bool = False):
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return len(self.params)

    def bind_param(self, param: md.Tensor):
        param.allow_grad = True
        self.params.append(param)


# wrapper to signify it's actually an activation
# better to do this because it requires the outputs from its corresponding layer, not the actual previous layer to compute the gradient
class ActivationLayer(Layer):
    def __init__(self, activation_function: mdt.UnaryOp):
        self.activation_function = activation_function

    def forward(self, x: md.Tensor) -> md.Tensor:
        return self.activation_function(x)


class Dense(OptimizableLayer):
    def __init__(self, n_neurons: int, n_weights: int, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.n_weights = n_weights
        self.weights = None
        self.biases = None

    def save_layer(self, fstream: BinaryIO):
        md.save(fstream, self.weights)
        md.save(fstream, self.biases)

    def load_layer(self, fstream: BinaryIO):
        self.weights = md.load(fstream)
        self.biases = md.load(fstream)

    # called when actually added to a net
    def setup(self, trainable: bool = True, reset_params: bool = False):
        self.trainable = trainable
        # store weights as matrix of n_weights x n_neurons
        # n_weights should be the same as n_neurons of the previous layer
        # pre-transposing here so we don't have to later for calculations
        # note: should probably change this rng later, but I don't think it actually matters that much
        if reset_params or self.weights is None:
            scale = math.sqrt(2 / self.n_weights)
            self.weights = scale * md.randn(self.n_weights, self.n_neurons)
            self.bind_param(self.weights)
        # store biases as row vector of size n_neurons
        if reset_params or self.biases is None:
            self.biases = md.zeros((1, self.n_neurons))
            self.bind_param(self.biases)

    # x should be a row vector
    # forward should return a row vector
    def forward(self, x: md.Tensor) -> md.Tensor:
        # number of inputs needs to match number of weights
        if x.shape[-1] != self.weights.shape[0]:
            raise ValueError(
                "Inputs to forward do not match the number of weights in this layer"
            )
        # btw this is not a dot product, it's a matrix multiplication
        # this is the actual linear combination. The matrix multiplication multiplies each weight
        # by its corresponding input, and returns a vector representing the outputs from each neuron
        # it is much faster to do one massive matrix multiplication here since a pure python loop would be
        # relatively incredibly slow compared to a GPU-optimized matrix multiplication
        product = md.matmul(x, self.weights) + self.biases
        return product


# the dropout layer is used to prevent the network from overfitting
# network will overfit when the neurons activate specifically to match the training dataset
# so, we can just randomly turn off some randomly selected portion of the neurons to disrupt that
# this way, the neurons will be forced to learn more robust/reliable methods of analyzing features
# from its inputs. similar to making students use different techniques to answer questions
# rather than memorizing the test itself
class Dropout(Layer):
    def __init__(self, prob: float, auto_scale: bool = True, trainable: bool = False):
        self.p = prob
        self.auto_scale = auto_scale
        self.trainable = trainable

    def forward(self, x: md.Tensor) -> md.Tensor:
        return F.dropout(
            x, self.p, auto_scale=self.auto_scale, trainable=self.trainable
        )


# it can be difficult for the individual layers of a network to learn
# if its inputs are kind of all over the place. the BatchNormalization layer
# attempts to normalize all the inputs throughout a batch to have a mean of 0
# and standard deviation of 1. This way, the layers have a more predictable spread
# of inputs, and should be able to learn quite a bit faster and be more confident (lower loss)
class BatchNormalization(OptimizableLayer):
    def __init__(
        self, n_dimensions: int, epsilon: float = 1e-3, momentum: float = 0.99, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_dimensions = n_dimensions
        self.epsilon = epsilon
        self.momentum = momentum
        self.moving_means = md.zeros(self.n_dimensions)
        self.moving_variances = md.ones(self.n_dimensions)

        self.gamma = None
        self.beta = None

    def save_layer(self, fstream: BinaryIO):
        md.save(fstream, self.gamma)
        md.save(fstream, self.beta)
        md.save(fstream, self.moving_means)
        md.save(fstream, self.moving_variances)

    def load_layer(self, fstream: BinaryIO):
        self.gamma = md.load(fstream)
        self.beta = md.load(fstream)
        self.moving_means = md.load(fstream)
        self.moving_variances = md.load(fstream)

    def setup(self, trainable: bool = True, reset_params: bool = False):
        self.trainable = trainable
        if reset_params or self.gamma is None:
            self.gamma = md.ones(self.n_dimensions)
            self.bind_param(self.gamma)
        if reset_params or self.beta is None:
            self.beta = md.zeros(self.n_dimensions)
            self.bind_param(self.beta)

    def forward(self, x: md.Tensor) -> md.Tensor:
        return F.batchnormalize(
            x,
            self.gamma,
            self.beta,
            epsilon=self.epsilon,
            momentum=self.momentum,
            trainable=self.trainable,
            moving_means=self.moving_means,
            moving_variances=self.moving_variances,
        )


# we just flatten from the in_shape to a row vector
class FlatteningLayer(Layer):
    def __init__(self, in_shape: Union[int, Sequence[int]]):
        self.flat_len = math.prod(in_shape)

    def forward(self, x: md.Tensor) -> md.Tensor:
        return x.reshape((-1, self.flat_len))


# opposite of FlatteningLayer
class ExpandingLayer(Layer):
    def __init__(self, out_shape: Union[int, Sequence[int]]):
        self.out_shape = out_shape

    def forward(self, x: md.Tensor) -> md.Tensor:
        return x.reshape((-1, *self.out_shape))


# Convolutional layers are extremely useful for image analysis.
# They take some kernel, dot product it with fixed sections of an input image,
# and output another image containing different image features
class Conv2D(OptimizableLayer):
    def __init__(
        self,
        in_height: int,
        in_width: int,
        in_channels: int,
        padding: Union[int, float, Tuple[int, int, int, int]] = 0,
        n_kernels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.padding = get_padded_edges(padding)
        self.n_kernels = n_kernels
        self.stride = stride
        self.in_channels = in_channels

        if isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
        else:
            self.kernel_height = self.kernel_width = kernel_size

        out_dims = calculate_convolved_dimensions(
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
        self.forward_indices = calculate_im2col_indices(
            *out_dims, self.kernel_height, self.kernel_width, self.stride
        )
        self.backward_input_indices = calculate_im2col_indices(
            in_height, in_width, self.kernel_height, self.kernel_width, self.stride
        )
        self.backward_kern_indices = calculate_im2col_indices(
            self.kernel_height, self.kernel_width, *out_dims, self.stride
        )

        self.kernels = None

    def save_layer(self, fstream: BinaryIO):
        md.save(fstream, self.kernels)

    def load_layer(self, fstream: BinaryIO):
        self.kernels = md.load(fstream)

    def setup(self, trainable: bool = True, reset_params: bool = False):
        self.trainable = trainable
        if reset_params or self.kernels is None:
            fan_in = self.kernel_height * self.kernel_width * self.in_channels
            scale = math.sqrt(2.0 / fan_in)
            self.kernels = scale * md.randn(
                self.n_kernels,
                self.kernel_height,
                self.kernel_width,
                self.in_channels,
            )
            self.bind_param(self.kernels)

    # accepts two dimensional image, outputs 2 dimensional image
    def forward(self, x: md.Tensor) -> md.Tensor:
        return F.convolve2d(
            x,
            self.kernels,
            padding=self.padding,
            stride=self.stride,
            forward_indices=self.forward_indices,
            backward_input_indices=self.backward_input_indices,
            backward_kern_indices=self.backward_kern_indices,
        )


class MaxPooling2D(Layer):
    def __init__(
        self,
        in_height: int,
        in_width: int,
        pool_size: int,
        stride: Optional[int] = None,
    ):
        self.pool_size = pool_size
        if stride is None:
            stride = pool_size
        self.stride = stride

        # self.in_dims = (in_height, in_width)
        out_dims = calculate_convolved_dimensions(
            in_height, in_width, self.pool_size, self.pool_size, 0, self.stride
        )

        self.forward_indices = calculate_im2col_indices(
            *out_dims, self.pool_size, self.pool_size, self.stride
        )

    def forward(self, x: md.Tensor) -> md.Tensor:
        return F.maxpool2d(
            x,
            self.pool_size,
            stride=self.stride,
            forward_indices=self.forward_indices,
        )


class MeanPooling2D(Layer):
    def __init__(
        self,
        in_height: int,
        in_width: int,
        pool_size: int,
        stride: Optional[int] = None,
    ):
        self.pool_size = pool_size
        if stride is None:
            stride = pool_size
        self.stride = stride

        # self.in_dims = (in_height, in_width)
        out_dims = calculate_convolved_dimensions(
            in_height, in_width, self.pool_size, self.pool_size, 0, self.stride
        )

        self.forward_indices = calculate_im2col_indices(
            *out_dims, self.pool_size, self.pool_size, self.stride
        )

    def forward(self, x: md.Tensor) -> md.Tensor:
        return F.meanpool2d(
            x,
            self.pool_size,
            stride=self.stride,
            forward_indices=self.forward_indices,
        )


class ResidualBlock(OptimizableLayer):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers = layers
        self.update_params()

    def update_params(self):
        for layer in self.layers:
            if not isinstance(layer, OptimizableLayer):
                continue
            self.params.extend(layer.params)

    def setup(self, trainable: bool = True, reset_params: bool = False):
        for layer in self.layers:
            if not isinstance(layer, OptimizableLayer):
                continue
            layer.setup(trainable=trainable, reset_params=reset_params)

    def save_layer(self, fstream: BinaryIO):
        for layer in self.layers:
            if isinstance(layer, OptimizableLayer):
                layer.save_layer(fstream)

    def load_layer(self, fstream: BinaryIO):
        for layer in self.layers:
            if isinstance(layer, OptimizableLayer):
                layer.load_layer(fstream)

    def forward(self, x: md.Tensor) -> md.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y + x
