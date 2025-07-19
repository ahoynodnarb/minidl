from __future__ import annotations

import minidiff as md

import minidl.functions as F


# there's a whole separate activation function class since some activations like leakyrelu take in kwargs that need to be held constant during training
# you can't really elegantly replicate passing those kwargs without using a class that wraps the underlying activation function
class ActivationFunction:
    def __call__(self, x: md.Tensor) -> md.Tensor:
        raise NotImplementedError


class Linear(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        return F.linear(x)


class Softmax(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        return F.softmax(x)


class ReLU(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        return F.relu(x)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x: md.Tensor) -> md.Tensor:
        return F.leakyrelu(x, alpha=self.alpha)


class Sigmoid(ActivationFunction):
    def __call__(self, x: md.Tensor) -> md.Tensor:
        return F.sigmoid(x)
