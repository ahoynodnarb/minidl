import minidiff as md


class LossFunction:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        raise NotImplementedError

    def total_correct(self, y_true, y_pred):
        raise NotImplementedError

    def accuracy(self, y_true, y_pred):
        raise NotImplementedError


class CrossEntropy(LossFunction):
    def __init__(self, precompute_grad=False, smoothing=0):
        self.precompute_grad = precompute_grad
        self.smoothing = smoothing

    # y_true should be a one-hot vector
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor):
        if y_true is None:
            raise ValueError("Empty ground truth array")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        n_classes = y_true.shape[-1]
        y_smoothed = (1 - self.smoothing) * y_true + (self.smoothing / n_classes)
        # avoid division by 0
        y_pred = y_pred.clip(1e-8, None)
        # compute the one hot loss, reshape to match
        loss = -md.sum(y_smoothed * md.log(y_pred), axis=-1, keepdims=True)
        return loss

    def gradient(self, y_true: md.Tensor, y_pred: md.Tensor):
        y_pred = y_pred.clip(1e-8, None)
        n_classes = y_true.shape[-1]
        y_smoothed = (1 - self.smoothing) * y_true + (self.smoothing / n_classes)
        # more numerically stable than -y_true / y_pred
        if self.precompute_grad:
            return (y_pred - y_smoothed) / len(y_smoothed)
        return -y_smoothed / y_pred

    def total_correct(self, y_true: md.Tensor, y_pred: md.Tensor):
        overlap = md.argmax(y_true, axis=-1) == md.argmax(y_pred, axis=-1)
        total_correct = md.sum(overlap)
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor):
        return self.total_correct(y_true, y_pred) / len(y_true)


class BinaryCrossEntropy(LossFunction):
    # y_true should be a one-hot vector
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor):
        # make sure y_true is one_hot
        if y_true is None:
            raise ValueError("Empty ground truth array")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        # avoid division by 0
        y_pred = y_pred.clip(1e-8, None)
        return -y_true * md.log(y_pred) - (1 - y_true) * md.log(1 - y_pred)

    def gradient(self, y_true: md.Tensor, y_pred: md.Tensor):
        y_pred = y_pred.clip(1e-8, None)
        # extra term discourages confidently bad results
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    def total_correct(self, y_true: md.Tensor, y_pred: md.Tensor):
        overlap = md.argmax(y_true, axis=-1) == md.argmax(y_pred, axis=-1)
        total_correct = md.sum(overlap)
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor):
        return self.total_correct(y_true, y_pred) / y_true


class MeanSquaredError(LossFunction):
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor):
        if y_true is None:
            raise ValueError("Empty ground truth array")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return md.mean(md.square(y_true - y_pred))

    def gradient(self, y_true: md.Tensor, y_pred: md.Tensor):
        grad = 2 * (y_pred - y_true) / y_true.shape[0]
        return grad

    def total_correct(self, y_true, y_pred, tolerance=0.1):
        return md.sum(md.abs(y_true - y_pred) < tolerance)

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor, tolerance=0.1):
        return self.total_correct(y_true, y_pred, tolerance=tolerance) / len(y_true)
