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
        return md.ones(x.shape)


class Softmax(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        # subtracting the maximum keeps the exponentiated values low
        # after doing the algebra, the results are the same
        mx = md.max(x, axis=-1, keepdims=True)
        exponentiated = md.exp(x - mx)
        return exponentiated / md.sum(exponentiated, axis=-1, keepdims=True)

    # computing the jacobian, used https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
    def gradient(self, x: md.Tensor) -> md.Tensor:
        # summing up each row of the jacobian just gives this
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
        return md.where(x >= 0, x, scaled)

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
