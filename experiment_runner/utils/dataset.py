import os
import pathlib
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
from keras.datasets import cifar10, mnist

from .image import load_image


def one_hot(label, num_classes):
    label_one_hot = np.zeros(num_classes)
    label_one_hot[label] = 1
    return label_one_hot


def stratified_sampling(labels, sample_percentage=0.01):
    l = defaultdict(list)
    for i, label in enumerate(labels):
        l[label].append(i)

    sample = []
    for label in l.values():
        n = int(np.ceil(len(label) * sample_percentage))
        sample.extend(np.random.choice(label, size=n, replace=False))
    return sample


class BaseDataset:
    def __init__(self, transform=None):
        self.transform = transform
        self._build_dataset()
        self._build_classes()

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index is out of range!")

        image = self.data["image"][index]
        label = self.data["label"][index]

        if isinstance(image, str):
            image = load_image(image)

        if isinstance(label, str):
            label = self.cls2idx[label]

        if self.transform is not None:
            image = self.transform(image)["image"]

        label = one_hot(label, self.num_classes)

        return image, label

    @property
    def num_classes(self):
        return len(self.classes)

    def class_distribution(self, normalize=True):
        count = dict(Counter(self.data["label"]))
        if normalize:
            count = {c: n / len(self) for c, n in count.items()}
        return {c: count[c] for c in sorted(count)}

    def _build_dataset(self):
        data = defaultdict(list)
        return data

    def _build_classes(self):
        if isinstance(self.data["label"], np.ndarray):
            self.data["label"] = self.data["label"].flatten()
        self.classes = sorted(list(set(self.data["label"])))
        self.cls2idx = {v: k for k, v in enumerate(self.classes)}

    def sample(self, sample_percentage):
        dataset = deepcopy(self)
        index = stratified_sampling(self.data["label"], sample_percentage)
        for key, value in dataset.data.items():
            dataset.data[key] = value[index]
        return dataset


class ImageFolder(BaseDataset):
    def __init__(self, root, transform=None):
        self.root = root
        super().__init__(transform)

    def _build_dataset(self):
        self.data = defaultdict(list)
        for path_cls in pathlib.Path(self.root).glob("*"):
            for path_img in path_cls.glob("*"):
                self.data["image"].append(os.path.normpath(path_img))
                self.data["label"].append(os.path.basename(path_cls))


class MNIST(BaseDataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        super().__init__(transform)

    def _build_dataset(self):
        subset = 0 if self.train else 1
        self.data = defaultdict(list)
        image, label = mnist.load_data()[subset]
        self.data["image"] = np.expand_dims(image, axis=-1)
        self.data["label"] = label


class CIFAR10(BaseDataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        super().__init__(transform)

    def _build_dataset(self):
        subset = 0 if self.train else 1
        self.data = defaultdict(list)
        image, label = cifar10.load_data()[subset]
        self.data["image"] = image
        self.data["label"] = label
