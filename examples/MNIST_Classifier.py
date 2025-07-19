import minidiff as md
from mnist import MNIST

from minidl.activation_functions import ReLU
from minidl.layers import (
    ActivationLayer,
    Conv2D,
    Dense,
    Dropout,
    ExpandingLayer,
    FlatteningLayer,
)
from minidl.loss_functions import CrossEntropy
from minidl.neural_networks import NeuralNetwork
from minidl.optimizers import Adam


# the labels are initially the actual number, which is not what we want
# we actually want a vector where the index of the correct output is 1, and the rest are 0
# like a probability distribution
def format_labels(labels):
    formatted = md.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        formatted[i][label] = 1
    return formatted


def train_network(network):
    image_data = MNIST("./examples/MNIST/")
    training_images, training_labels = image_data.load_training()
    # normalize the images to be between 0 and 1, so the inputs are not too massive
    # then shuffle so we aren't always training with the exact same dataset
    training_images = md.Tensor(training_images) / 255.0
    training_labels = md.Tensor(training_labels)

    training_labels = format_labels(training_labels)

    network.trainable = True
    network.train(training_images, training_labels, batch_size=64, epochs=50)


def test_network(network):
    data = MNIST("./examples/MNIST/")

    testing_images, testing_labels = data.load_testing()
    testing_images = md.Tensor(testing_images) / 255.0

    testing_labels = format_labels(md.Tensor(testing_labels))
    network.trainable = False
    network.test(testing_images, testing_labels, batch_size=64)


def test_dataset_at_index(network, index):
    import cv2

    data = MNIST("./examples/MNIST/")
    testing_images, testing_labels = data.load_testing()

    test_image = md.Tensor(testing_images[index])
    test_label = testing_labels[index]

    pred = network(test_image / 255.0)
    print(f"prediction: {md.argmax(pred)}")
    print(f"actual: {test_label}")

    cv_image = test_image.astype(md.uint8).reshape((28, 28))
    cv2.imshow(f"MNIST Test Dataset entry {index}", cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    network = NeuralNetwork(
        loss_function=CrossEntropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
    )
    network.set_layers(
        ExpandingLayer((28, 28, 1)),
        Conv2D(28, 28, 1, padding=2, n_kernels=5, kernel_size=5, stride=1),
        Dropout(0.25),
        ActivationLayer(ReLU()),
        FlatteningLayer((28, 28, 5)),
        Dense(256, 28 * 28 * 5),
        ActivationLayer(ReLU()),
        Dense(10, 256),
    )
    try:
        train_network(network)
    except KeyboardInterrupt:
        pass
    finally:
        ans = input("Save? y/n")
        if ans == "y":
            network.save_network("./examples/MNIST_Classifier.npy")
