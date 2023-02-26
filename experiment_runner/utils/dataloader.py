from collections.abc import Iterable, Mapping

import numpy as np
import tensorflow as tf

__all__ = ["DataLoader"]


def default_collate(batch):
    batch_size = len(batch)
    if batch_size <= 0:
        raise ValueError("Batch is empty!")

    if isinstance(batch[0], Mapping):
        keys = batch[0].keys()
        return {
            key: np.array([batch[i][key] for i in range(batch_size)]) for key in keys
        }

    if isinstance(batch[0], Iterable):
        value_count = len(batch[0])
        return tuple(
            np.array([batch[i][j] for i in range(batch_size)])
            for j in range(value_count)
        )

    return np.array([batch[i] for i in range(batch_size)])


class DataLoader(tf.keras.utils.Sequence):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=default_collate,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.order = np.arange(len(self.dataset))

    def __len__(self):
        num_samples = len(self.dataset)
        remainder = num_samples % self.batch_size
        total_batch = (num_samples - remainder) // self.batch_size
        if remainder > 0 and not self.drop_last:
            total_batch += 1
        return total_batch

    def __getitem__(self, index):
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

    def on_epoch_end(self):
        if self.shuffle:
            self.order = np.random.permutation(self.order)
