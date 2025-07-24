from __future__ import annotations

from typing import TYPE_CHECKING

import minidiff as md

if TYPE_CHECKING:
    from typing import List


def split_batches(data: md.Tensor, batch_size: int) -> List[md.Tensor]:
    if batch_size == 1:
        return md.expand_dims(data, axis=1)
    indices = range(batch_size, len(data), batch_size)
    return md.split(data, indices)


def shuffle_dataset(data: md.Tensor, labels: md.Tensor) -> tuple[md.Tensor, md.Tensor]:
    indices = md.permutation(len(data))
    return (data[indices], labels[indices])
