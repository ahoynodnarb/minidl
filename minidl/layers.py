try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import math


def add_padding(mat, padding=None):
    batch_size, height, width, channels = mat.shape

    if (
        padding is None
        or (isinstance(padding, tuple) and sum(padding) == 0)
        or padding == 0
    ):
        return mat

    if isinstance(padding, tuple):
        pad_top, pad_bottom, pad_left, pad_right = padding
    else:
        if padding % 1 == 0:
            pad_top = pad_bottom = pad_left = pad_right = padding
        else:
            padding = int(math.floor(padding))
            pad_top = pad_left = padding
            pad_bottom = pad_right = padding + 1

    padded = np.zeros(
        (
            batch_size,
            pad_top + height + pad_bottom,
            pad_left + width + pad_right,
            channels,
        )
    )

    padded[:, pad_top : height + pad_top, pad_left : width + pad_left, :] = mat
    return padded


def calculate_same_padding(height, width, kernel_height, kernel_width, stride):
    pad_vert = (height * (stride - 1) + kernel_height - stride) / 2
    pad_hori = (width * (stride - 1) + kernel_width - stride) / 2

    if pad_vert % 1 == 0:
        pad_top = pad_bottom = pad_vert
    else:
        pad_vert = int(math.floor(pad_vert))
        pad_top, pad_bottom = pad_vert, pad_vert + 1

    if pad_hori % 1 == 0:
        pad_left = pad_right = pad_hori
    else:
        pad_hori = int(math.floor(pad_hori))
        pad_left, pad_right = pad_hori, pad_hori + 1

    return (pad_top, pad_bottom, pad_left, pad_right)


def calculate_full_padding(kernel_height, kernel_width, original_padding):
    pad_top = kernel_height - 1
    pad_bottom = kernel_height - 1
    pad_left = kernel_width - 1
    pad_right = kernel_width - 1
    if isinstance(original_padding, tuple):
        o_top, o_bottom, o_left, o_right = original_padding
        pad_top -= o_top
        pad_bottom -= o_bottom
        pad_left -= o_left
        pad_right -= o_right
    else:
        pad_top -= original_padding
        pad_bottom -= original_padding
        pad_left -= original_padding
        pad_right -= original_padding
    return (pad_top, pad_bottom, pad_left, pad_right)


def calculate_convolved_dimensions(
    height, width, kernel_height, kernel_width, padding, stride
):
    if isinstance(padding, tuple):
        top, bottom, left, right = padding
        vertical_padding = top + bottom
        horizontal_padding = left + right
    else:
        vertical_padding = int(2 * padding)
        horizontal_padding = int(2 * padding)

    out_height = (height - kernel_height + vertical_padding) // stride + 1
    out_width = (width - kernel_width + horizontal_padding) // stride + 1
    return (out_height, out_width)


class Parameter:
    def __init__(self, default=None):
        self.default_val = default
        self.grad_func = None

    def __set_name__(self, obj, name):
        self.internal_name = "_" + name
        self.__set__(obj, self.default_val)

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.internal_name)

    def __set__(self, obj, value):
        setattr(obj, self.internal_name, value)

    def grad(self, func):
        self.grad_func = func
        return func


class Layer:
    trainable: bool = False

    def forward(self, inputs):
        raise NotImplementedError

    # need to know gradient to compute gradient of cost function with respect to weights, biases, and outputs
    # need previous outputs (the outputs from the previous/left layer) to compute gradient of cost function with respect to weights
    # each of these should be a row vector
    # grad of shape (n_batches, n_neurons)
    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)


class OptimizableLayer(Layer):
    def __init__(self, l2_lambda=0, clip=None):
        self.l2_lambda = l2_lambda
        self.clip = clip
        self.grad_funcs = []
        self.bind_grads()

    def save_layer(self, fstream):
        raise NotImplementedError

    def load_layer(self, fstream):
        raise NotImplementedError

    def setup(self, trainable=True):
        raise NotImplementedError

    @property
    def n_params(self):
        return len(self.grad_funcs)

    def bind_grads(self):
        class_dict = self.__class__.__dict__
        for field in class_dict.values():
            if not isinstance(field, Parameter):
                continue
            self.grad_funcs.append((field, field.grad_func))

    def update_params(self, grad, optimizers):
        for (param, func), optimizer in zip(self.grad_funcs, optimizers):
            param_val = param.__get__(self)
            grad_wrt_param = func(self, grad)
            param_updated = optimizer.update(
                param_val, grad_wrt_param, l2_lambda=self.l2_lambda
            )
            param.__set__(self, param_updated)


# wrapper to signify it's actually an activation
# better to do this because it requires the outputs from its corresponding layer, not the actual previous layer to compute the gradient
class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def forward(self, inputs):
        self.prev_outputs = inputs
        return self.activation_function(inputs)

    def backward(self, grad):
        return grad * self.activation_function.gradient(self.prev_outputs)


class Dense(OptimizableLayer):
    weights = Parameter()
    biases = Parameter()

    def __init__(self, n_neurons: int, n_weights: int, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.n_weights = n_weights

    def save_layer(self, fstream):
        np.save(fstream, self.weights)
        np.save(fstream, self.biases)

    def load_layer(self, fstream):
        self.weights = np.load(fstream)
        self.biases = np.load(fstream)

    def compute_grad_wrt_x(self, grad):
        return grad.dot(self.weights.T)

    @weights.grad
    def compute_grad_wrt_w(self, grad):
        return self.prev_outputs.T.dot(grad)

    @biases.grad
    def compute_grad_wrt_biases(self, grad):
        return np.sum(grad, axis=0, keepdims=True)

    # called when actually added to a net
    def setup(self, trainable=True):
        self.trainable = trainable
        # store weights as matrix of n_weights x n_neurons
        # n_weights should be the same as n_neurons of the previous layer
        # pre-transposing here so we don't have to later for calculations
        # note: should probably change this rng later, but I don't think it actually matters that much
        if self.weights is None:
            scale = np.sqrt(2 / self.n_weights)
            self.weights = scale * np.random.randn(self.n_weights, self.n_neurons)
        # store biases as row vector of size n_neurons
        if self.biases is None:
            self.biases = np.zeros((1, self.n_neurons))

    # inputs should be a row vector
    # forward should return a row vector
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # number of inputs needs to match number of weights
        if inputs.shape[-1] != self.weights.shape[0]:
            raise ValueError(
                "Inputs to forward do not match the number of weights in this layer"
            )
        # btw this is not a dot product, it's a matrix multiplication
        # this is the actual linear combination. The matrix multiplication multiplies each weight
        # by its corresponding input, and returns a vector representing the outputs from each neuron
        # it is much faster to do one massive matrix multiplication here since a pure python loop would be
        # relatively incredibly slow compared to a GPU-optimized matrix multiplication
        product = inputs.dot(self.weights) + self.biases
        if self.trainable:
            self.prev_outputs = inputs
        return product

    # gradient should be a row vector (will represent the derivative of the cost function with respect to this layer's outputs)
    # gradient will be of shape (n_batches, n_neurons)
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if grad.shape[-1] != self.weights.shape[-1]:
            raise ValueError(
                "Number of gradients in grad does not match the number of neurons in this layer"
            )
        # to compute the gradient of the cost function with respect to the preceding layer's neurons:
        # you multiply the gradient of the cost function with respect to this layer's outputs
        # by the gradient of this layer's outputs with respect to the preceding layer's outputs
        # for each neuron in the preceding layer, that gradient is just the sum of the weights since those are what
        # directly cause change in this layer's outputs
        return self.compute_grad_wrt_x(grad)


# the dropout layer is used to prevent the network from overfitting
# network will overfit when the neurons activate specifically to match the training dataset
# so, we can just randomly turn off some randomly selected portion of the neurons to disrupt that
# this way, the neurons will be forced to learn more robust/reliable methods of analyzing features
# from its inputs. similar to making students use different techniques to answer questions
# rather than memorizing the test itself
class Dropout(Layer):
    def __init__(self, prob: float, auto_scale=True, trainable=False):
        self.p = prob
        self.auto_scale = auto_scale
        self.trainable = trainable

        self.mask = None

    def forward(self, inputs):
        if not self.trainable:
            return inputs
        self.mask = np.random.binomial(1, 1 - self.p, inputs.shape)
        if self.auto_scale:
            return self.mask * inputs / (1 - self.p)
        return self.mask * inputs

    def backward(self, grad):
        if not self.trainable:
            return grad
        if self.auto_scale:
            return self.mask * grad / (1 - self.p)
        return self.mask * grad


# WIP
class LayerNormalization(OptimizableLayer):
    weights = Parameter()
    biases = Parameter()

    def __init__(self, n_dimensions, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.n_dimensions = n_dimensions
        self.epsilon = epsilon

    def compute_grad_wrt_x(self, grad):
        # Gradient through scaling (w) and normalization (z)
        grad_z = grad * self.weights  # dy/dz = w

        # Gradient through variance (σ²)
        grad_var = (
            np.sum(grad_z * self.shifted, axis=-1, keepdims=True)
            * (-0.5)
            * (self.inverse_std**3)
        )

        # Gradient through mean (μ)
        grad_mean = np.sum(grad_z, axis=-1, keepdims=True) * (
            -self.inverse_std
        ) + grad_var * np.mean(-2 * self.shifted, axis=-1, keepdims=True)

        # Combine gradients
        grad_wrt_x = (
            (grad_z * self.inverse_std)
            + (1 / self.n_dimensions) * grad_var * 2 * self.shifted
            + (1 / self.n_dimensions) * grad_mean
        )
        return grad_wrt_x

    @weights.grad
    def compute_grad_wrt_w(self, grad):
        summed_dims = tuple(range(grad.ndim - 1))
        return np.sum(self.norm * grad, axis=summed_dims, keepdims=True)

    @biases.grad
    def compute_grad_wrt_biases(self, grad):
        summed_dims = tuple(range(grad.ndim - 1))
        return np.sum(grad, axis=summed_dims, keepdims=True)

    def setup(self, trainable=True):
        self.trainable = trainable
        if self.weights is None:
            self.weights = np.ones(self.n_dimensions)
        if self.biases is None:
            self.biases = np.zeros(self.n_dimensions)

    # https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    def forward(self, inputs):
        if self.trainable:
            self.prev_outputs = inputs
        means = np.mean(inputs, axis=-1, keepdims=True)
        self.shifted = inputs - means
        self.variance = np.sum(self.shifted**2, axis=-1, keepdims=True) + self.epsilon
        self.inverse_std = self.variance**-0.5
        self.norm = self.shifted * self.inverse_std
        return self.norm * self.weights + self.biases

    def backward(self, grad):
        grad_wrt_x = self.compute_grad_wrt_x(grad)
        return grad_wrt_x


# it can be difficult for the individual layers of a network to learn
# if its inputs are kind of all over the place. the BatchNormalization layer
# attempts to normalize all the inputs throughout a batch to have a mean of 0
# and standard deviation of 1. This way, the layers have a more predictable spread
# of inputs, and should be able to learn quite a bit faster and be more confident (lower loss)
class BatchNormalization(OptimizableLayer):
    gamma = Parameter()
    beta = Parameter()

    def __init__(self, n_dimensions, epsilon=1e-3, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.n_dimensions = n_dimensions
        self.epsilon = epsilon
        self.momentum = momentum
        self.moving_means = np.zeros(self.n_dimensions)
        self.moving_variances = np.ones(self.n_dimensions)

    def compute_grad_wrt_x(self, grad):
        ndims = len(grad.shape)
        norm_axes = tuple(range(ndims - 1))  # (0, 1, 2) for images
        m = math.prod([grad.shape[axis] for axis in norm_axes])

        gamma_reshaped = self.gamma.reshape((1,) * (ndims - 1) + (-1,))

        # dL/dx_hat
        grad_x_hat = grad * gamma_reshaped

        # dL/dvar
        grad_var = np.sum(
            grad_x_hat * self.mean_deviation * -0.5 * (self.std_deviation**-3),
            axis=norm_axes,
            keepdims=True,
        )

        # dL/dmean
        term1 = np.sum(-grad_x_hat / self.std_deviation, axis=norm_axes, keepdims=True)
        term2 = (
            grad_var
            * np.sum(-2 * self.mean_deviation, axis=norm_axes, keepdims=True)
            / m
        )
        grad_mean = term1 + term2

        # dL/dx components
        component1 = grad_x_hat / self.std_deviation
        component2 = grad_var * 2 * self.mean_deviation / m
        component3 = grad_mean / m

        grad_input = component1 + component2 + component3

        return grad_input

    @gamma.grad
    def compute_grad_wrt_gamma(self, grad):
        normalized_dimensions = tuple(range(grad.ndim - 1))
        return np.sum(grad * self.x_hat, axis=normalized_dimensions)

    @beta.grad
    def compute_grad_wrt_beta(self, grad):
        normalized_dimensions = tuple(range(grad.ndim - 1))
        return np.sum(grad, axis=normalized_dimensions)

    def save_layer(self, fstream):
        np.save(fstream, self.gamma)
        np.save(fstream, self.beta)
        np.save(fstream, self.moving_means)
        np.save(fstream, self.moving_variances)

    def load_layer(self, fstream):
        self.gamma = np.load(fstream)
        self.beta = np.load(fstream)
        self.moving_means = np.load(fstream)
        self.moving_variances = np.load(fstream)

    def setup(self, trainable=True):
        self.trainable = trainable
        if self.gamma is None:
            self.gamma = np.ones(self.n_dimensions)
        if self.beta is None:
            self.beta = np.zeros(self.n_dimensions)

    def forward(self, inputs):
        normalized_dimensions = tuple(range(inputs.ndim - 1))
        if self.trainable:
            self.prev_outputs = inputs
        dummy_dims = [1] * (len(inputs.shape) - 1)
        gamma_reshaped = self.gamma.reshape((*dummy_dims, self.n_dimensions))
        beta_reshaped = self.beta.reshape((*dummy_dims, self.n_dimensions))

        if not self.trainable:
            means_reshaped = self.moving_means.reshape((*dummy_dims, self.n_dimensions))
            variances_reshaped = self.moving_variances.reshape(
                (*dummy_dims, self.n_dimensions)
            )

            normalized = (inputs - means_reshaped) / np.sqrt(
                variances_reshaped + self.epsilon
            )

            return normalized * gamma_reshaped + beta_reshaped

        self.means = np.mean(
            inputs, axis=normalized_dimensions, keepdims=True
        )  # returns mu for each dimension of input
        self.mean_deviation = inputs - self.means
        self.variances = np.mean(
            np.square(self.mean_deviation), axis=normalized_dimensions, keepdims=True
        )  # returns sigma^2 for each input
        self.std_deviation = np.sqrt(self.variances + self.epsilon)

        self.x_hat = self.mean_deviation / self.std_deviation

        means_flat = self.means.reshape(-1)
        variances_flat = self.variances.reshape(-1)

        self.moving_means = self.moving_means * self.momentum + means_flat * (
            1 - self.momentum
        )
        self.moving_variances = (
            self.moving_variances * self.momentum + variances_flat * (1 - self.momentum)
        )
        return gamma_reshaped * self.x_hat + beta_reshaped

    # grad is a row vector of (n_batches, n_dimensions of previous layer)
    def backward(self, grad):
        # gradient of the outputs of this layer with respect to the outputs of the last layer
        return self.compute_grad_wrt_x(grad)


# we just flatten from the in_shape to a row vector
class FlatteningLayer(Layer):
    def __init__(self, in_shape):
        self.in_shape = in_shape
        self.flat_len = math.prod(self.in_shape)

    def forward(self, inputs):
        # elements per batch
        return inputs.reshape((-1, self.flat_len))

    def backward(self, grad):
        grad = grad.reshape((-1, *self.in_shape))
        return grad


# opposite of FlatteningLayer
class ExpandingLayer(Layer):
    def __init__(self, out_shape):
        self.out_shape = out_shape
        self.flat_len = math.prod(self.out_shape)

    def forward(self, inputs):
        return inputs.reshape((-1, *self.out_shape))

    def backward(self, grad):
        return grad.reshape((-1, self.flat_len))


# assuming batch and dims includes kernels
# rather than actually convolving the matrix and kernel, we flatten the matrix into another matrix
# where every column is made up of the elements in each patch of the input matrix
# then we flatten the kernels into a row vector so we can actually perform the matrix multiplication


def perform_convolution(
    mat, kernels, padding=None, stride=1, im2col_indices=None, out_dims=None, out=None
):
    orig_shape = mat.shape
    batch_size, orig_height, orig_width, _ = orig_shape
    n_kernels, kernel_height, kernel_width, kernel_channels = kernels.shape

    if out_dims is None:
        out_dims = calculate_convolved_dimensions(
            orig_height, orig_width, kernel_height, kernel_width, padding, stride
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
    convolved = np.tensordot(as_cols, flattened_kernels, axes=((1, 3), (1, 2)))
    reshaped = convolved.reshape(out_shape)
    return reshaped


def calculate_im2col_indices(rows_out, cols_out, kernel_height, kernel_width, stride):
    # these are the indices that correspond to each row within the patch
    kernel_row_indices = np.repeat(np.arange(kernel_height), kernel_width)
    # these are the indices corresponding to the row portion of the position of each patch within the input matrix
    conv_row_indices = stride * np.repeat(np.arange(rows_out), cols_out)

    # these are the indices that correspond to each column within the patch
    kernel_col_indices = np.tile(np.arange(kernel_width), kernel_height)
    # these are the indices that correspond to the column portion of the position of each patch within the input matrix
    conv_col_indices = stride * np.tile(np.arange(cols_out), rows_out)

    row_indices = kernel_row_indices.reshape((-1, 1)) + conv_row_indices.reshape(
        (1, -1)
    )
    col_indices = kernel_col_indices.reshape((-1, 1)) + conv_col_indices.reshape(
        (1, -1)
    )

    return (row_indices, col_indices)


# Convolutional layers are extremely useful for image analysis.
# They take some kernel, dot product it with fixed sections of an input image,
# and output another image containing different image features
class Conv2D(OptimizableLayer):
    kernels = Parameter()
    biases = Parameter()

    def __init__(
        self,
        in_height,
        in_width,
        in_channels,
        padding=0,
        n_kernels=1,
        kernel_size=3,
        stride=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(kernel_size, tuple):
            kernel_height, kernel_width = kernel_size
        else:
            kernel_height = kernel_width = kernel_size

        if isinstance(padding, tuple):
            pad_top, pad_bottom, pad_left, pad_right = padding
        else:
            pad_top = pad_bottom = pad_left = pad_right = padding
        if (in_height - kernel_height + pad_top + pad_bottom) % stride != 0:
            raise ValueError("Cannot evenly convolve")
        if (in_width - kernel_width + pad_left + pad_right) % stride != 0:
            raise ValueError("Cannot evenly convolve")

        # we need to keep track of the shape of the inputs and outputs so we do not
        # have to recalculate them for every single batch
        self.in_channels = in_channels
        self.in_dims = (in_height, in_width)
        self.out_dims = calculate_convolved_dimensions(
            in_height, in_width, kernel_height, kernel_width, padding, stride
        )

        self.padding = padding
        self.n_kernels = n_kernels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride

        # we optimize the actual convolution as a large matrix multiplication
        # and we keep track of how the matrices need to be rearranged for that
        # matrix multiplication, also so we don't have to recompute it for each batch
        self.forward_indices = calculate_im2col_indices(
            *self.out_dims, self.kernel_height, self.kernel_width, self.stride
        )
        self.backward_input_indices = calculate_im2col_indices(
            *self.in_dims, self.kernel_height, self.kernel_width, self.stride
        )
        self.backward_kern_indices = calculate_im2col_indices(
            self.kernel_height, self.kernel_width, *self.out_dims, self.stride
        )

    # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
    # we compute the gradient with respect to the layer's inputs by rotating the kernels diagonally, transposing the kernel channel
    # and number of kernel dimension of the kernel matrix, and convolving the gradient by that new kernel matrix
    def compute_grad_wrt_x(self, grad):
        # rotate kernels, then swap axes to match up correctly
        flipped_kernels = np.flip(np.flip(self.kernels, axis=1), axis=2)
        flipped_kernels = np.swapaxes(flipped_kernels, -1, 0)

        full_padding = calculate_full_padding(
            kernel_height=self.kernel_width,
            kernel_width=self.kernel_height,
            original_padding=self.padding,
        )
        grad_wrt_x = perform_convolution(
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
    @kernels.grad
    def compute_grad_wrt_w(self, grad):
        # normally, computing grad_wrt_w requires you to do convolutions for each slice of the previous outputs
        # and each slice of the gradient. But we can take advantage of batching to instead treat each slice of
        # output as a separate entry to the batch, and each slice of the gradient as a separate "kernel"
        # this results in us having the same final convolution, just the slices end up as the channels instead
        swapped_prev_outputs = np.swapaxes(self.prev_outputs, 0, -1)
        swapped_grad = np.swapaxes(grad, 0, -1)
        convolved = perform_convolution(
            swapped_prev_outputs,
            swapped_grad,
            padding=self.padding,
            stride=self.stride,
            out_dims=(self.kernel_height, self.kernel_width),
            im2col_indices=self.backward_kern_indices,
        )
        grad_wrt_w = np.swapaxes(convolved, 0, -1)

        return grad_wrt_w

    @biases.grad
    def compute_grad_wrt_biases(self, grad):
        return np.sum(grad, axis=0, keepdims=True)

    def save_layer(self, fstream):
        np.save(fstream, self.kernels)
        np.save(fstream, self.biases)

    def load_layer(self, fstream):
        self.kernels = np.load(fstream)
        self.biases = np.load(fstream)

    def setup(self, trainable=True):
        self.trainable = trainable
        if self.kernels is None:
            fan_in = self.kernel_height * self.kernel_width * self.in_channels
            scale = np.sqrt(2.0 / fan_in)
            self.kernels = scale * np.random.randn(
                self.n_kernels,
                self.kernel_height,
                self.kernel_width,
                self.in_channels,
            )
        if self.biases is None:
            self.biases = np.zeros(self.n_kernels)

    # accepts two dimensional image, outputs 2 dimensional image
    def forward(self, inputs):
        if self.trainable:
            self.prev_outputs = inputs
        convolved = perform_convolution(
            inputs,
            self.kernels,
            padding=self.padding,
            stride=self.stride,
            out_dims=self.out_dims,
            im2col_indices=self.forward_indices,
        )
        return convolved + self.biases

    # assumes the gradient has the same shape as out_shape
    def backward(self, grad):
        return self.compute_grad_wrt_x(grad)


class MaxPooling2D(Layer):
    def __init__(
        self,
        in_height,
        in_width,
        in_channels,
        pool_size,
        stride=None,
    ):
        if stride is None:
            stride = pool_size

        self.in_dims = (in_height, in_width)
        self.out_dims = calculate_convolved_dimensions(
            in_height, in_width, pool_size, pool_size, 0, stride
        )

        self.in_channels = in_channels
        self.pool_size = pool_size
        self.stride = stride

        self.forward_indices = calculate_im2col_indices(
            *self.out_dims, pool_size, pool_size, stride
        )

        self.row_offset, self.col_offset = self.precompute_window_offsets()

    # this entire function essentially just computes the top left corner of each patch
    def precompute_window_offsets(self):
        out_height, out_width = self.out_dims

        # number of pools is just how many can fit in within the cropped area
        n_pools = out_height * out_width

        # this is just 0,1,2,...n_pools - 1. It is just the position of each pool
        pool_indices = np.arange(n_pools)[..., np.newaxis]

        # finally the actual offsets.
        # window_rows represents the row of the top left corner of the pool
        # window_cols represents the column of the top left corner of the pool
        window_rows = (pool_indices // out_width) * self.stride
        window_cols = (pool_indices % out_width) * self.stride

        return (window_rows, window_cols)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        out_shape = (batch_size, *self.out_dims, self.in_channels)
        self.prev_inputs = inputs

        row_indices, col_indices = self.forward_indices

        # transform inputs into column vectors of patches
        as_cols = inputs[:, row_indices, col_indices, :]

        flat_indices = np.argmax(as_cols, axis=1, keepdims=True)

        # add precomputed offsets to the indices since flat_indices gives coordinates relative to individual patches.
        # but we need indices relative to the entire input matrix
        row_max_indices = flat_indices // self.pool_size + self.row_offset
        col_max_indices = flat_indices % self.pool_size + self.col_offset

        # need these indices as placeholders to correctly index
        batch_indices = np.arange(batch_size)[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]  # shape: (batch_size, 1, 1, 1)
        channel_indices = np.arange(self.in_channels)[
            np.newaxis, np.newaxis, np.newaxis, ...
        ]  # shape: (1, 1, 1, in_channels)

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

    def backward(self, grad):
        batch_size = grad.shape[0]
        in_shape = (batch_size, *self.in_dims, self.in_channels)

        batch_indices, row_max_indices, col_max_indices, channel_indices = (
            self.prev_indices
        )

        # the gradient for every element that is not the max is zeroed out, this is kind of
        # like a blank canvas before we paint on the gradients
        zeros = np.zeros(in_shape)
        flattened_grad = grad.reshape((batch_size, 1, -1, self.in_channels))

        # similar to forward function, just assigning at these indices instead
        # this is the "painting" step
        zeros[batch_indices, row_max_indices, col_max_indices, channel_indices] = (
            flattened_grad
        )

        return zeros


class MeanPooling2D(Layer):
    def __init__(
        self,
        in_height,
        in_width,
        in_channels,
        pool_size,
        stride=None,
    ):
        if stride is None:
            stride = pool_size

        self.in_dims = (in_height, in_width)
        self.out_dims = calculate_convolved_dimensions(
            in_height, in_width, pool_size, pool_size, 0, stride
        )

        self.in_channels = in_channels
        self.pool_size = pool_size
        self.stride = stride

        self.forward_indices = calculate_im2col_indices(
            *self.out_dims, pool_size, pool_size, stride
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        out_shape = (batch_size, *self.out_dims, self.in_channels)
        self.prev_inputs = inputs

        row_indices, col_indices = self.forward_indices

        # transform inputs into column vectors of patches
        as_cols = inputs[:, row_indices, col_indices, :]

        averaged = np.mean(as_cols, axis=1, keepdims=True)
        return averaged.reshape(out_shape)

    def backward(self, grad):
        batch_size, grad_height, grad_width, _ = grad.shape
        # add the extra 1 so that indexing broadcasts for the entire patch
        flattened_grad = grad.reshape(
            (batch_size, 1, grad_height * grad_width, self.in_channels)
        )
        grad_wrt_x = np.zeros((batch_size, *self.in_dims, self.in_channels))
        row_indices, col_indices = self.forward_indices
        grad_wrt_x[:, row_indices, col_indices, :] = flattened_grad / (
            self.pool_size * self.pool_size
        )
        return grad_wrt_x


class ResidualBlock(OptimizableLayer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        self.update_grad_funcs()

    # optimizers will be in the correct order
    def update_params(self, grad, optimizers):
        i = 0
        for x, layer in enumerate(self.layers):
            if not isinstance(layer, OptimizableLayer):
                continue
            n_params = layer.n_params
            layer_optimizers = optimizers[i : i + n_params]
            layer_grad = self.intermediate_gradients[x]
            layer.update_params(layer_grad, layer_optimizers)
            i += layer.n_params

    def update_grad_funcs(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, OptimizableLayer):
                self.grad_funcs.extend(
                    [(param, func) for param, func in layer.grad_funcs]
                )

    def setup(self, trainable=True):
        for layer in self.layers:
            if not isinstance(layer, OptimizableLayer):
                continue
            layer.setup(trainable=trainable)

    def save_layer(self, fstream):
        for layer in self.layers:
            if isinstance(layer, OptimizableLayer):
                layer.save_layer(fstream)

    def load_layer(self, fstream):
        for layer in self.layers:
            if isinstance(layer, OptimizableLayer):
                layer.load_layer(fstream)

    def forward(self, inputs):
        y = inputs
        for layer in self.layers:
            y = layer(y)
        return y + inputs

    def backward(self, grad):
        accum_grad = grad
        self.intermediate_gradients = []
        for layer in reversed(self.layers):
            self.intermediate_gradients.insert(0, accum_grad)
            accum_grad = layer.backward(accum_grad)
        return accum_grad + grad
