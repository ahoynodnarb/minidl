import albumentations as A
import minidiff as md

from minidl.activation_functions import ReLU
from minidl.layers import (
    ActivationLayer,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    FlatteningLayer,
    MaxPooling2D,
)
from minidl.loss_functions import CrossEntropy
from minidl.neural_networks import NeuralNetwork
from minidl.optimizers import AdamW, ReduceLROnPlateau

ONE_HOT_TO_PLAINTEXT = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEANS = md.Tensor([0.49118418, 0.4825434, 0.44775498])
CIFAR10_STDS = md.Tensor([0.2464881, 0.24259564, 0.26116827])


def load_formatted_data(filenames):
    def unpickle(fn):
        import pickle

        with open(fn, "rb") as file:
            dct = pickle.load(file, encoding="bytes")
        return dct

    data = {}
    for filename in filenames:
        print(filename)
        file_data = unpickle(filename)
        data.update(file_data)

    def convert_to_one_hot(labels, classes):
        n_labels = len(labels)
        one_hot = md.zeros((n_labels, classes))
        one_hot[md.arange(n_labels), labels] = 1
        return one_hot

    def format_images(images):
        images = md.Tensor(images) / 255.0
        channel_length = 32 * 32
        red = images[:, :channel_length]
        green = images[:, channel_length : channel_length * 2]
        blue = images[:, channel_length * 2 : channel_length * 3]
        channeled = md.stack((red, green, blue), axis=-1)
        reshaped = channeled.reshape((-1, 32, 32, 3))
        return reshaped

    images = format_images(data[b"data"])
    labels = convert_to_one_hot(data[b"labels"], 10)

    return (images, labels)


def load_meta():
    def unpickle(fn):
        import pickle

        with open(fn, "rb") as file:
            dct = pickle.load(file, encoding="bytes")
        return dct

    out = {}
    file_data = unpickle("./examples/cifar-10-batches-py/batches.meta")
    out.update(file_data)
    return out


def normalize_data(data):
    return (data - CIFAR10_MEANS) / CIFAR10_STDS


def test(network: NeuralNetwork):
    import os

    parent_directory = "./examples/cifar-10-batches-py/"
    testing_images, testing_labels = load_formatted_data(
        [os.path.join(parent_directory, "test_batch")]
    )
    network.test(
        testing_images, testing_labels, batch_size=64, norm_func=normalize_data
    )


def train(network: NeuralNetwork):
    import os

    parent_directory = "./examples/cifar-10-batches-py/"
    data_batch_files = [
        os.path.join(parent_directory, f"data_batch_{x + 1}") for x in range(5)
    ]

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                },
            ),
        ]
    )

    def aug_func(images: md.Tensor) -> md.Tensor:
        transformed_data = transform(images=images.as_numpy())
        transformed_images = transformed_data["images"]
        return md.Tensor(transformed_images)

    training_images, training_labels = load_formatted_data(data_batch_files)

    testing_images, testing_labels = load_formatted_data(
        [os.path.join(parent_directory, "test_batch")]
    )
    network.trainable = True
    network.train(
        training_images,
        training_labels,
        batch_size=64,
        epochs=50,
        norm_func=normalize_data,
        aug_func=aug_func,
        val_data=testing_images,
        val_labels=testing_labels,
    )


if __name__ == "__main__":
    optim = AdamW(learning_rate=5e-4)
    scheduler = ReduceLROnPlateau(optimizer=optim, factor=0.5, patience=3, min_lr=1e-6)
    # probably don't change dropout, increasing it gets too aggressive and the network stops learning midway through
    cross_entropy = CrossEntropy(from_logits=True, smoothing=0.05)
    network = NeuralNetwork(loss_function=cross_entropy, optimizer=scheduler)
    network.set_layers(
        Conv2D(
            32, 32, 3, padding=1, n_kernels=32, kernel_size=3, stride=1, l2_lambda=1e-4
        ),
        ActivationLayer(ReLU()),
        BatchNormalization(32),
        Conv2D(
            32, 32, 32, padding=1, n_kernels=32, kernel_size=3, stride=1, l2_lambda=1e-4
        ),
        ActivationLayer(ReLU()),
        BatchNormalization(32),
        MaxPooling2D(32, 32, pool_size=2),
        Dropout(0.25),
        Conv2D(
            16, 16, 32, padding=1, n_kernels=64, kernel_size=3, stride=1, l2_lambda=1e-4
        ),
        ActivationLayer(ReLU()),
        BatchNormalization(64),
        Conv2D(
            16, 16, 64, padding=1, n_kernels=64, kernel_size=3, stride=1, l2_lambda=1e-4
        ),
        ActivationLayer(ReLU()),
        BatchNormalization(64),
        MaxPooling2D(16, 16, pool_size=2),
        Dropout(0.25),
        Conv2D(
            8, 8, 64, padding=1, n_kernels=128, kernel_size=3, stride=1, l2_lambda=1e-4
        ),
        ActivationLayer(ReLU()),
        BatchNormalization(128),
        Conv2D(
            8, 8, 128, padding=1, n_kernels=128, kernel_size=3, stride=1, l2_lambda=1e-4
        ),
        ActivationLayer(ReLU()),
        BatchNormalization(128),
        MaxPooling2D(8, 8, pool_size=2),
        Dropout(0.25),
        FlatteningLayer((4, 4, 128)),
        Dense(128, 4 * 4 * 128, l2_lambda=1e-4),
        ActivationLayer(ReLU()),
        BatchNormalization(128),
        Dense(10, 128, l2_lambda=1e-4),
    )
    try:
        train(network)
    except KeyboardInterrupt:
        pass
    finally:
        ans = input("Save? y/n")
        if ans == "y":
            network.save_network("./examples/cifar10.npy")
