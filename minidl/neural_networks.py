from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import minidiff as md
from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Callable, List, Optional, Tuple

    from minidl.layers import Layer, Optimizer

from minidl.layers import OptimizableLayer
from minidl.loss_functions import LossFunction
from minidl.optimizers import LRScheduler
from minidl.utils.data import shuffle_dataset, split_batches


class NeuralNetwork:
    def __init__(
        self, loss_function: LossFunction, optimizer: Optimizer, trainable: bool = False
    ):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._layers: List[Layer] = []
        self._trainable = trainable
        self.layers_setup = False
        self.layer_optimizers: List[Optimizer] = []

    def __call__(self, inputs: md.Tensor) -> md.Tensor:
        return self.feed_forward(inputs)

    def save_network(self, filename):
        with open(filename, "wb") as f:
            for layer in self.layers:
                if not hasattr(layer, "save_layer"):
                    continue
                layer.save_layer(f)

    def load_network(self, filename):
        with open(filename, "rb") as f:
            for layer in self.layers:
                if not hasattr(layer, "load_layer"):
                    continue
                layer.load_layer(f)

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, trainable: bool):
        self._trainable = trainable
        for layer in self.layers:
            layer.trainable = trainable

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        self._layers = []
        for layer in layers:
            layer.trainable = self.trainable
            self._layers.append(layer)

    def setup_layers(self, force: bool = False):
        with md.no_grad():
            for layer in self.layers:
                should_setup = not self.layers_setup or force
                if not should_setup or not isinstance(layer, OptimizableLayer):
                    continue

                layer.setup(trainable=self.trainable)

                optimizers = []
                for _ in range(layer.n_params):
                    new_optimizer = deepcopy(self.optimizer)
                    optimizers.append(new_optimizer)
                self.layer_optimizers.append(optimizers)

        self.layers_setup = True

    def set_layers(self, *layers: Layer):
        self.layers = layers

    def feed_forward(self, inputs: md.Tensor) -> md.Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def update_layer_weights(self):
        optimizeable_layers = [
            layer for layer in self.layers if isinstance(layer, OptimizableLayer)
        ]
        for optimizers, layer in zip(self.layer_optimizers, optimizeable_layers):
            if not layer.trainable:
                continue
            for optimizer, param in zip(optimizers, layer.params):
                optimizer.update(param)
                param.grad = None

    def train(
        self,
        data: md.Tensor,
        labels: md.Tensor,
        batch_size: int = 1,
        epochs: int = 1,
        val_data: Optional[md.Tensor] = None,
        val_labels: Optional[md.Tensor] = None,
        norm_func: Callable[[md.Tensor], md.Tensor] = None,
        aug_func: Callable[[md.Tensor], md.Tensor] = None,
        print_output: bool = True,
    ):
        self.trainable = True
        self.setup_layers()

        train_data = data
        train_labels = labels

        if val_data is None or val_labels is None:
            val_len = int(len(train_data) * 0.15)
            indices = md.permutation(len(train_data))
            train_indices = indices[val_len:]
            val_indices = indices[:val_len]
            train_data, train_labels = data[train_indices], labels[train_indices]
            val_data, val_labels = data[val_indices], labels[val_indices]

        for epoch in range(epochs):
            self.trainable = True

            shuffled_train_data, shuffled_train_labels = shuffle_dataset(
                train_data, train_labels
            )
            if aug_func is not None:
                shuffled_train_data = aug_func(shuffled_train_data)

            if norm_func is not None:
                shuffled_train_data = norm_func(shuffled_train_data)

            batched_train_data = split_batches(shuffled_train_data, batch_size)
            batched_train_labels = split_batches(shuffled_train_labels, batch_size)

            dataset = zip(batched_train_data, batched_train_labels)
            if print_output:
                dataset = tqdm(
                    dataset,
                    desc=f"Epoch #{epoch + 1}",
                    total=len(batched_train_data),
                    bar_format="{l_bar}{bar:30}{r_bar}",
                )

            total_training_correct = 0
            total_training_loss = 0

            for x, y_true in dataset:
                y_pred = self(x)

                loss = self.loss_function(y_true, y_pred)
                loss.backward(cleanup_mode="destroy")

                with md.no_grad():
                    self.update_layer_weights()
                    total_training_correct += self.loss_function.total_correct(
                        y_true, y_pred
                    )
                    total_training_loss += md.sum(loss).item()

            val_acc, avg_val_loss = self.test(
                val_data,
                val_labels,
                batch_size=batch_size,
                norm_func=norm_func,
                print_output=False,
            )

            training_acc = total_training_correct / len(train_data)
            avg_training_loss = total_training_loss / len(train_data)

            if isinstance(self.optimizer, LRScheduler):
                for optimizers in self.layer_optimizers:
                    for optimizer in optimizers:
                        optimizer.update_state(epoch, avg_val_loss)

            if print_output:
                print(
                    f"Validation accuracy at the end of epoch {epoch + 1}: {val_acc}",
                    sep="\n",
                )
                print(
                    f"Average validation loss at the end of epoch {epoch + 1}: {avg_val_loss}",
                    sep="\n",
                )
                print(
                    f"Training accuracy at the end of epoch {epoch + 1}: {training_acc}",
                    sep="\n",
                )
                print(
                    f"Average training loss at the end of epoch {epoch + 1}: {avg_training_loss}",
                    sep="\n",
                )

    def test(
        self,
        testing_data: md.Tensor,
        testing_labels: md.Tensor,
        batch_size: int = 1,
        norm_func: Callable[[md.Tensor], md.Tensor] = None,
        print_output: bool = True,
    ) -> Tuple[float, float]:
        self.trainable = False

        if norm_func is not None:
            testing_data = norm_func(testing_data)

        batched_testing_data = split_batches(testing_data, batch_size)
        batched_testing_labels = split_batches(testing_labels, batch_size)

        dataset = zip(batched_testing_data, batched_testing_labels)
        if print_output:
            dataset = tqdm(
                dataset,
                desc="Testing",
                total=len(batched_testing_data),
                bar_format="{l_bar}{bar:30}{r_bar}",
            )

        total_testing_correct = 0
        total_testing_loss = 0

        with md.no_grad():
            for x, y_true in dataset:
                y_pred = self(x)

                total_testing_correct += self.loss_function.total_correct(
                    y_true, y_pred
                )
                total_testing_loss += md.sum(self.loss_function(y_true, y_pred)).item()

        acc = total_testing_correct / len(testing_data)
        avg_loss = total_testing_loss / len(testing_data)

        if print_output:
            print(
                f"Total testing accuracy: {acc}",
                sep="\n",
            )
            print(
                f"Average testing loss: {avg_loss}",
                sep="\n",
            )

        return acc, avg_loss
