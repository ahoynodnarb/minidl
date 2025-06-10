try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import math


class ActivationFunction:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # should return the gradient of the function evaluated at each of the variables
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.ones(shape=x.shape)


class Softmax(ActivationFunction):
    def __init__(self, precompute_grad=False):
        self.precompute_grad = precompute_grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # subtracting the maximum keeps the exponentiated values low
        # after doing the algebra, the results are the same
        mx = np.max(x, axis=-1, keepdims=True)
        exponentiated = np.exp(x - mx)
        return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)

    # computing the jacobian, used https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # summing up each row of the jacobian just gives this
        if self.precompute_grad:
            return np.ones(x.shape)
        values = self(x)
        return (values * (1 - values)).clip(1e-8, None)


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0, None)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.alpha * x, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, self.alpha)


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        out = self(x)
        return out * (1 - out)


class GeLU(ActivationFunction):
    def __call__(self, x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    def gradient(self, x):
        return math.sqrt(2 / math.pi) * 3 * 0.044715 * x**2
