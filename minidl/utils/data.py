try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np


def split_batches(data, batch_size):
    if batch_size == 1:
        return np.expand_dims(data, axis=1)
    indices = range(batch_size, len(data), batch_size)
    return np.split(data, indices)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return sample_correct(y_true, y_pred) / len(y_true)


def sample_correct(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    overlap = np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)
    total_correct = np.sum(overlap)
    return total_correct


def shuffle_dataset(
    data: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return (data[indices], labels[indices])
