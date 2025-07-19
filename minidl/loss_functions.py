from __future__ import annotations

import minidiff as md

import minidl.functions as F


class LossFunction:
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
        raise NotImplementedError

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
        return F.cross_entropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            smoothing=self.smoothing,
        )

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

    def total_correct(self, y_true: md.Tensor, y_pred: md.Tensor) -> int:
        overlap = md.argmax(y_true, axis=-1) == md.argmax(y_pred, axis=-1)
        total_correct = md.sum(overlap).item()
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor) -> float:
        return self.total_correct(y_true, y_pred) / len(y_true)


class MeanSquaredError(LossFunction):
    def __call__(self, y_true: md.Tensor, y_pred: md.Tensor) -> md.Tensor:
        return F.mean_squared_error(y_true, y_pred)

    def total_correct(
        self, y_true: md.Tensor, y_pred: md.Tensor, tolerance: float = 0.1
    ) -> int:
        overlap = md.abs(y_true - y_pred) < tolerance
        total_correct = md.sum(overlap).item()
        return total_correct

    def accuracy(self, y_true: md.Tensor, y_pred: md.Tensor, tolerance=0.1) -> float:
        return self.total_correct(y_true, y_pred, tolerance=tolerance) / len(y_true)
