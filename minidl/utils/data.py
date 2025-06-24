import minidiff as md


def split_batches(data, batch_size):
    if batch_size == 1:
        return md.expand_dims(data, axis=1)
    indices = range(batch_size, len(data), batch_size)
    return md.split(data, indices)


def accuracy(y_true: md.Tensor, y_pred: md.Tensor) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return sample_correct(y_true, y_pred) / len(y_true)


def sample_correct(y_true: md.Tensor, y_pred: md.Tensor) -> int:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    overlap = md.argmax(y_true, axis=-1) == md.argmax(y_pred, axis=-1)
    total_correct = md.sum(overlap)
    return total_correct


def shuffle_dataset(data: md.Tensor, labels: md.Tensor) -> tuple[md.Tensor, md.Tensor]:
    indices = md.arange(len(data))
    md.shuffle(indices)
    return (data[indices], labels[indices])
