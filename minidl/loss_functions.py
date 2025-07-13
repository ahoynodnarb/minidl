import minidiff as md
import minidl.functions as F


class LossFunction:
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
        raise NotImplementedError

    # def gradient(self, y_true, y_pred):
    #     raise NotImplementedError

    def total_correct(self, y_true: md.Tensor, y_pred: md.Tensor) -> int:
        raise NotImplementedError

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor) -> float:
        raise NotImplementedError


class CrossEntropy(LossFunction):
    def __init__(self, from_logits: bool = False, smoothing: float = 0):
        self.from_logits = from_logits
        self.smoothing = smoothing

    # y_true should be a one-hot vector
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
        return F.cross_entropy(y_true, y_pred)
        # if y_true is None:
        #     raise ValueError("Empty ground truth array")
        # if y_true.shape != y_pred.shape:
        #     raise ValueError("y_true and y_pred must have the same shape")
        # n_classes = y_true.shape[-1]
        # y_smoothed = (1 - self.smoothing) * y_true + (self.smoothing / n_classes)
        # if self.from_logits:
        #     mx = md.max(y_pred, axis=-1, keepdims=True)
        #     e = md.exp(y_pred - mx)
        #     s = md.sum(e, axis=-1, keepdims=True)
        #     # log-sum-exp take the log of the sum of the exponents shifted by the max, and then shift again later
        #     lse = mx + md.log(s)
        #     return -md.sum(y_smoothed * (y_pred - lse), axis=-1, keepdims=True)
        # # avoid division by 0
        # y_pred = y_pred.clip(1e-8, None)
        # # compute the one hot loss, reshape to match
        # loss = -md.sum(y_smoothed * md.log(y_pred), axis=-1, keepdims=True)
        # return loss

    # def gradient(self, y_true: md.Tensor, y_pred: md.Tensor):
    #     y_pred = y_pred.clip(1e-8, None)
    #     n_classes = y_true.shape[-1]
    #     y_smoothed = (1 - self.smoothing) * y_true + (self.smoothing / n_classes)
    #     # more numerically stable than -y_true / y_pred
    #     if self.from_logits:
    #         return (y_pred - y_smoothed) / len(y_smoothed)
    #     return -y_smoothed / y_pred

    def total_correct(self, y_true: md.Tensor, y_pred: md.Tensor) -> int:
        overlap = md.argmax(y_true, axis=-1) == md.argmax(y_pred, axis=-1)
        total_correct = md.sum(overlap).item()
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor) -> float:
        return self.total_correct(y_true, y_pred) / len(y_true)


class BinaryCrossEntropy(LossFunction):
    # y_true should be a one-hot vector
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
        return F.binary_cross_entropy(y_true, y_pred)
        # # make sure y_true is one_hot
        # if y_true is None:
        #     raise ValueError("Empty ground truth array")
        # if y_true.shape != y_pred.shape:
        #     raise ValueError("y_true and y_pred must have the same shape")
        # # avoid division by 0
        # y_pred = y_pred.clip(1e-8, None)
        # return -md.mean(
        #     y_true * md.log(y_pred) + (1 - y_true) * md.log(1 - y_pred),
        #     axis=-1,
        #     keepdims=True,
        # )

    # def gradient(self, y_true: md.Tensor, y_pred: md.Tensor):
    #     y_pred = y_pred.clip(1e-8, None)
    #     # extra term discourages confidently bad results
    #     return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    def total_correct(self, y_true: md.Tensor, y_pred: md.Tensor) -> int:
        overlap = md.argmax(y_true, axis=-1) == md.argmax(y_pred, axis=-1)
        total_correct = md.sum(overlap).item()
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor) -> float:
        return self.total_correct(y_true, y_pred) / len(y_true)


class MeanSquaredError(LossFunction):
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
        return F.mean_squared_error(y_true, y_pred)
        # if y_true is None:
        #     raise ValueError("Empty ground truth array")
        # if y_true.shape != y_pred.shape:
        #     raise ValueError("y_true and y_pred must have the same shape")
        # averaged_dims = tuple(range(1, y_true.ndim))
        # return md.mean(md.square(y_true - y_pred), axis=averaged_dims)

    # def gradient(self, y_true: md.Tensor, y_pred: md.Tensor):
    #     averaged_dims = tuple(range(1, y_true.ndim))
    #     averaged_features = math.prod([y_true.shape[dim] for dim in averaged_dims])
    #     grad = 2 * (y_pred - y_true) / averaged_features
    #     return grad

    def total_correct(self, y_true, y_pred, tolerance=0.1) -> int:
        overlap = md.abs(y_true - y_pred) < tolerance
        total_correct = md.sum(overlap).item()
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor, tolerance=0.1) -> float:
        return self.total_correct(y_true, y_pred, tolerance=tolerance) / len(y_true)
