import math

import minidiff as md


class ActivationFunction:
    def __call__(self, x: md.Tensor) -> md.Tensor:
        raise NotImplementedError

    # should return the gradient of the function evaluated at each of the variables
    def gradient(self, x: md.Tensor) -> md.Tensor:
        raise NotImplementedError


class Linear(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        return x

    def gradient(self, x: md.Tensor) -> md.Tensor:
        return md.ones(shape=x.shape)


class Softmax(ActivationFunction):
    def __init__(self, use_logsoftmax=False):
        self.use_logsoftmax = use_logsoftmax

    def __call__(self, x: md.Tensor) -> md.Tensor:
        # subtracting the maximum keeps the exponentiated values low
        # after doing the algebra, the results are the same
        mx = md.max(x, axis=-1, keepdims=True)
        exponentiated = md.exp(x - mx)
        return exponentiated / md.sum(exponentiated, axis=-1, keepdims=True)

    # computing the jacobian, used https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
    def gradient(self, x: md.Tensor) -> md.Tensor:
        # summing up each row of the jacobian just gives this
        if self.use_logsoftmax:
            return md.ones(x.shape)
        values = self(x)
        return (values * (1 - values)).clip(1e-8, None)


class ReLU(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        return x.clip(0, None)

    def gradient(self, x: md.Tensor) -> md.Tensor:
        return md.where(x >= 0, 1, 0)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x: md.Tensor) -> md.Tensor:
        scaled = self.alpha * x
        return md.where(scaled >= x, scaled, x)

    def gradient(self, x: md.Tensor) -> md.Tensor:
        return md.where(x >= 0, 1, self.alpha)


class Sigmoid(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        x = x.clip(-500, 500)
        return 1 / (1 + md.exp(-x))

    def gradient(self, x: md.Tensor) -> md.Tensor:
        x = x.clip(-500, 500)
        out = self(x)
        return out * (1 - out)


class GeLU(ActivationFunction):
    def __call__(self, x):
        return 0.5 * x * (1 + md.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    def gradient(self, x):
        return math.sqrt(2 / math.pi) * 3 * 0.044715 * x**2
