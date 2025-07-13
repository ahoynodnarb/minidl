from copy import deepcopy

import minidiff as md
from tqdm import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

from minidl.layers import OptimizableLayer, Layer
from minidl.loss_functions import LossFunction
from minidl.optimizers import LRScheduler, Optimizer
from minidl.utils.data import shuffle_dataset, split_batches


class NeuralNetwork:
    def __init__(
        self, loss_function: LossFunction, optimizer: Optimizer, trainable=False
    ):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._layers: List[Layer] = []
        self._trainable = trainable
        self.layers_setup = False
        self.layer_optimizers: List[Optimizer] = []

    def __call__(self, inputs):
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
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, trainable):
        self._trainable = trainable
        for layer in self.layers:
            layer.trainable = trainable

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = []
        for layer in layers:
            layer.trainable = self.trainable
            self._layers.append(layer)

    def setup_layers(self, force=False):
        for layer in self.layers:
            should_setup = self.layers_setup and not force
            if should_setup or not isinstance(layer, OptimizableLayer):
                continue

            optimizers = []
            for _ in range(layer.n_params):
                new_optimizer = deepcopy(self.optimizer)
                optimizers.append(new_optimizer)

            self.layer_optimizers.append(optimizers)
            layer.setup(trainable=self.trainable)

        self.layers_setup = True

    def set_layers(self, *layers):
        self.layers = layers

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    # def backpropagate(self, grad):
    #     optimizer_idx = 0
    #     for layer in reversed(self.layers):
    #         grad = grad.clip(-1.0, 1.0)
    #         old_grad = grad
    #         grad = layer.backward(grad)
    #         if not layer.trainable:
    #             continue
    #         if not isinstance(layer, OptimizableLayer):
    #             continue
    #         optimizers = self.layer_optimizers[optimizer_idx]
    #         layer.update_params(old_grad, optimizers)
    #         optimizer_idx += 1

    #     return grad

    def update_layer_weights(self):
        for layer_optimizers, layer in zip(self.layer_optimizers, self.layers):
            if not isinstance(layer, OptimizableLayer):
                continue
            if not layer.trainable:
                continue
            for layer_optimizer, param in zip(layer_optimizers, layer.params):
                layer_optimizer.update(param)

            # layer_optimizers = self.layer_optimizers

    def train(
        self,
        data,
        labels,
        batch_size=1,
        epochs=1,
        val_data=None,
        val_labels=None,
        norm_func=None,
        aug_func=None,
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

        if norm_func is not None:
            val_data = norm_func(val_data)

        batched_val_data = split_batches(val_data, batch_size)
        batched_val_labels = split_batches(val_labels, batch_size)

        for epoch in range(epochs):
            shuffled_train_data, shuffled_train_labels = shuffle_dataset(
                train_data, train_labels
            )
            if aug_func is not None:
                shuffled_train_data = aug_func(shuffled_train_data)

            if norm_func is not None:
                shuffled_train_data = norm_func(shuffled_train_data)

            batched_train_data = split_batches(shuffled_train_data, batch_size)
            batched_train_labels = split_batches(shuffled_train_labels, batch_size)

            self.trainable = True
            total_training_correct = 0
            total_training_loss = 0

            progress = tqdm(
                zip(batched_train_data, batched_train_labels),
                total=len(batched_train_data),
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            progress.set_description(f"Epoch #{epoch + 1}")
            self.check = False
            for x, y_true in progress:
                y_pred = self(x)
                loss = self.loss_function(y_true, y_pred)
                loss.backward()

                with md.no_grad():
                    self.update_layer_weights()
                    total_training_correct += self.loss_function.total_correct(
                        y_true, y_pred
                    )
                    total_training_loss += md.sum(loss)
                # grad = self.loss_function.gradient(y_true, y_pred)
                # self.backpropagate(grad)

            self.trainable = False
            total_val_correct = 0
            total_val_loss = 0
            for x, y_true in zip(batched_val_data, batched_val_labels):
                val_pred = self(x)
                total_val_correct += self.loss_function.total_correct(y_true, val_pred)
                total_val_loss += md.sum(self.loss_function(y_true, val_pred))

            val_acc = total_val_correct / len(val_data)
            avg_val_loss = total_val_loss / len(val_data)

            training_acc = total_training_correct / len(train_data)
            avg_training_loss = total_training_loss / len(train_data)

            if isinstance(self.optimizer, LRScheduler):
                for optimizers in self.layer_optimizers:
                    for optimizer in optimizers:
                        optimizer.update_state(epoch, avg_val_loss)

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

    def test(self, testing_data, testing_labels, batch_size=1, norm_func=None):
        self.trainable = False

        if norm_func is not None:
            testing_data = norm_func(testing_data)

        batched_testing_data = split_batches(testing_data, batch_size)
        batched_testing_labels = split_batches(testing_labels, batch_size)

        total_testing_correct = 0
        total_testing_loss = 0

        progress = tqdm(
            zip(batched_testing_data, batched_testing_labels),
            total=len(batched_testing_data),
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        for x, y_true in progress:
            progress.set_description("Testing...")
            y_pred = self(x)

            total_testing_correct += self.loss_function.total_correct(y_true, y_pred)
            total_testing_loss += md.sum(self.loss_function(y_true, y_pred))

        print(
            f"Total testing accuracy: {total_testing_correct / len(testing_data)}",
            sep="\n",
        )
        print(
            f"Average testing loss: {total_testing_loss / len(testing_data)}", sep="\n"
        )
