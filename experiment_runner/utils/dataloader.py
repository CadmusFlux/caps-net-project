from typing import Dict, Any, Union, Tuple, Callable, Optional, Sequence, List, Mapping

import numpy as np
import tensorflow as tf

__all__ = ["DataLoader"]


def default_collate(batch: List[Union[Dict[Any, np.ndarray], Tuple[np.ndarray, ...], np.ndarray]]) \
        -> Union[Mapping[Any, np.ndarray], Sequence[np.ndarray, ...], np.ndarray]:
    batch_size = len(batch)
    if batch_size <= 0:
        raise ValueError("Batch is empty!")

    if isinstance(batch[0], Mapping):
        keys = batch[0].keys()
        return {
            key: np.array([batch[i][key] for i in range(batch_size)]) for key in keys
        }

    if isinstance(batch[0], Sequence):
        value_count = len(batch[0])
        return tuple(
            np.array([batch[i][j] for i in range(batch_size)])
            for j in range(value_count)
        )

    return np.array([batch[i] for i in range(batch_size)])


class DataLoader(tf.keras.utils.Sequence):
    def __init__(
            self,
            dataset: Sequence,
            batch_size: int,
            shuffle: Optional[bool] = False,
            drop_last: Optional[bool] = False,
            collate_fn: Optional[Callable] = default_collate,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.order = np.arange(len(self.dataset))

    def __len__(self) -> int:
        num_samples = len(self.dataset)
        remainder = num_samples % self.batch_size
        total_batch = (num_samples - remainder) // self.batch_size
        if remainder > 0 and not self.drop_last:
            total_batch += 1
        return total_batch

    def __getitem__(self, index: int) -> Union[Dict[Any, np.ndarray], Tuple[np.ndarray, ...], np.ndarray]:
        if index >= len(self):
            raise IndexError("Index is out of range!")

        batch = []

        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        for i in range(start, stop):
            if i >= len(self.dataset):
                break
            batch.append(self.dataset[self.order[i]])

        return self.collate_fn(batch)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.order = np.random.permutation(self.order)
