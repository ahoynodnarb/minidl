try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

from layers import (
    Dense,
    Conv2D,
    LayerNormalization,
    BatchNormalization,
    OptimizableLayer,
)

from tqdm import tqdm


def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient of function f at point x
    f: function that takes x and returns scalar loss
    x: point to compute gradient at
    h: small step size
    """
    grad = np.zeros_like(x)
    flat_x = x.flatten()
    flat_grad = grad.flatten()

    prog = tqdm(range(len(flat_x)))
    prog.set_description("Evaluating numerical gradients...")
    for i in prog:
        # f(x + h)
        flat_x[i] += h
        x_plus = flat_x.reshape(x.shape)
        loss_plus = f(x_plus)

        # f(x - h)
        flat_x[i] -= 2 * h
        x_minus = flat_x.reshape(x.shape)
        loss_minus = f(x_minus)

        # numerical gradient
        flat_grad[i] = (loss_plus - loss_minus) / (2 * h)

        # restore original value
        flat_x[i] += h

    return flat_grad.reshape(x.shape)


def test_grad_wrt_parameter(
    layer: OptimizableLayer,
    test_input,
    target_output,
    grad_wrt_param,
    loss_fun,
    param,
    param_name,
):
    layer.trainable = True

    print(f"Testing gradient w.r.t. {param_name}...")

    # print("forward")
    output = layer.forward(test_input)
    # print("done")
    grad_output = output - target_output

    analytical_grad_param = grad_wrt_param(grad_output)
    numerical_grad_param = numerical_gradient(loss_fun, param)

    layer.trainable = False

    # Compare
    diff_param = np.abs(analytical_grad_param - numerical_grad_param)
    max_diff_param = np.max(diff_param)
    relative_error_param = max_diff_param / (
        np.max(np.abs(numerical_grad_param)) + 1e-8
    )

    print(f"Max absolute difference ({param_name}): {max_diff_param:.2e}")
    print(f"Relative error ({param_name}): {relative_error_param:.2e}")

    if relative_error_param < 1e-5:
        print(f"✅ {param_name} gradients look correct!")
    else:
        print(f"❌ {param_name} gradients may have issues")
        print("Analytical grad sample:", analytical_grad_param.flatten()[:5])
        print("Numerical grad sample:", numerical_grad_param.flatten()[:5])


def test_dense_gradients():
    """Test your Dense implementation against numerical gradients"""
    print("Testing dense...")

    # Create a tiny Dense layer for testing
    dense = Dense(32, 64)

    # Initialize with small random weights for stability
    dense.weights = 0.1 * np.random.randn(64, 32)
    dense.biases = 0.1 * np.random.randn(1, 32)

    # Create small test input and target
    batch_size = 10
    test_input = np.random.randn(batch_size, 64)
    target_output = np.random.randn(batch_size, 32)

    def loss_function_weights(weights):
        """Loss as function of weights"""
        # conv.kernels = kernels
        dense.weights = weights
        output = dense.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_biases(biases):
        """Loss as a function of biases"""
        dense.biases = biases
        output = dense.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_inputs(input_data):
        """Loss as function of input"""
        output = dense.forward(input_data)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    # weights
    test_grad_wrt_parameter(
        layer=dense,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=dense.compute_grad_wrt_w,
        loss_fun=loss_function_weights,
        param=dense.weights,
        param_name="weights",
    )
    print("\n")
    # biases
    test_grad_wrt_parameter(
        layer=dense,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=dense.compute_grad_wrt_biases,
        loss_fun=loss_function_biases,
        param=dense.biases,
        param_name="biases",
    )
    print("\n")
    # inputs
    test_grad_wrt_parameter(
        layer=dense,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=dense.compute_grad_wrt_x,
        loss_fun=loss_function_inputs,
        param=test_input,
        param_name="inputs",
    )
    print("\n")


def test_conv2d_gradients():
    """Test your Conv2D implementation against numerical gradients"""

    print("Testing Conv2D...")

    # Create a tiny Conv2D layer for testing
    conv = Conv2D(
        in_height=4,
        in_width=4,
        in_channels=2,
        n_kernels=3,
        kernel_size=3,
        padding=1,
        stride=1,
    )

    # Initialize with small random weights for stability
    conv.kernels = 0.1 * np.random.randn(3, 3, 3, 2)  # (n_kernels, h, w, in_channels)
    conv.biases = 0.1 * np.random.randn(4, 4, 3)  # (out_h, out_w, n_kernels)

    # Create small test input and target
    batch_size = 10
    test_input = np.random.randn(batch_size, 4, 4, 2)
    target_output = np.random.randn(batch_size, 4, 4, 3)

    def loss_function_kernels(kernels):
        """Loss as function of kernel weights"""
        conv.kernels = kernels
        output = conv.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_biases(biases):
        """Loss as function of kernel biases"""
        conv.biases = biases
        output = conv.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_inputs(input_data):
        """Loss as function of input"""
        output = conv.forward(input_data)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    # kernels
    test_grad_wrt_parameter(
        layer=conv,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=conv.compute_grad_wrt_w,
        loss_fun=loss_function_kernels,
        param=conv.kernels,
        param_name="kernels",
    )
    print("\n")
    # biases
    test_grad_wrt_parameter(
        layer=conv,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=conv.compute_grad_wrt_biases,
        loss_fun=loss_function_biases,
        param=conv.biases,
        param_name="biases",
    )
    print("\n")
    # inputs
    test_grad_wrt_parameter(
        layer=conv,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=conv.compute_grad_wrt_x,
        loss_fun=loss_function_inputs,
        param=test_input,
        param_name="inputs",
    )
    print("\n")


def test_batchnorm_gradients():

    batch_size = 10
    n_dimensions = 16
    batchnorm = BatchNormalization(n_dimensions)
    batchnorm.gamma = 0.1 * np.random.randn(n_dimensions)
    batchnorm.beta = 0.1 * np.random.randn(n_dimensions)

    test_input = np.random.randn(batch_size, 10, 10, 16)
    target_output = np.random.randn(batch_size, 10, 10, 16)

    def loss_function_gamma(gamma):
        """Loss as function of batchnorm gammas"""
        batchnorm.gamma = gamma
        output = batchnorm.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_beta(beta):
        """Loss as function of batchnorm betas"""
        batchnorm.beta = beta
        output = batchnorm.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_inputs(input_data):
        """Loss as function of input"""
        # print("loss called")
        output = batchnorm.forward(input_data)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    # gamma
    test_grad_wrt_parameter(
        layer=batchnorm,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=batchnorm.compute_grad_wrt_gamma,
        loss_fun=loss_function_gamma,
        param=batchnorm.gamma,
        param_name="gammas",
    )
    print("\n")
    # beta
    test_grad_wrt_parameter(
        layer=batchnorm,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=batchnorm.compute_grad_wrt_beta,
        loss_fun=loss_function_beta,
        param=batchnorm.beta,
        param_name="betas",
    )
    print("\n")
    # inputs
    test_grad_wrt_parameter(
        layer=batchnorm,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=batchnorm.compute_grad_wrt_x,
        loss_fun=loss_function_inputs,
        param=test_input,
        param_name="inputs",
    )
    print("\n")


def test_layernorm_gradients():
    batch_size = 10
    n_dimensions = 16
    layernorm = LayerNormalization(n_dimensions)
    layernorm.weights = 0.1 * np.random.randn(n_dimensions)
    layernorm.biases = 0.1 * np.random.randn(n_dimensions)

    test_input = np.random.randn(batch_size, 10, 10, 16)
    target_output = np.random.randn(batch_size, 10, 10, 16)

    def loss_function_weights(weights):
        """Loss as function of batchnorm gammas"""
        layernorm.weights = weights
        output = layernorm.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_biases(biases):
        """Loss as function of batchnorm betas"""
        layernorm.biases = biases
        output = layernorm.forward(test_input)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    def loss_function_inputs(input_data):
        """Loss as function of input"""
        # print("loss called")
        output = layernorm.forward(input_data)
        loss = np.sum((output - target_output) ** 2) / 2
        return loss

    # gamma
    test_grad_wrt_parameter(
        layer=layernorm,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=layernorm.compute_grad_wrt_w,
        loss_fun=loss_function_weights,
        param=layernorm.weights,
        param_name="weights",
    )
    print("\n")
    # beta
    test_grad_wrt_parameter(
        layer=layernorm,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=layernorm.compute_grad_wrt_biases,
        loss_fun=loss_function_biases,
        param=layernorm.biases,
        param_name="biases",
    )
    print("\n")
    # inputs
    test_grad_wrt_parameter(
        layer=layernorm,
        test_input=test_input,
        target_output=target_output,
        grad_wrt_param=layernorm.compute_grad_wrt_x,
        loss_fun=loss_function_inputs,
        param=test_input,
        param_name="inputs",
    )
    print("\n")


if __name__ == "__main__":
    # Run the simple test first
    # simple_gradient_check()
    # print("\n")

    # Then run the full test
    test_conv2d_gradients()
    test_dense_gradients()
    test_layernorm_gradients()
    test_batchnorm_gradients()
