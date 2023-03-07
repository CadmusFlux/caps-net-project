from typing import Any, Dict, List, Union

import albumentations as A
import tensorflow_addons as tfa

from utils.layer import *

CIFAR10 = dict(
    size=(32, 32), mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), pad=8
)

MNIST = dict(size=(28, 28), mean=0.1307, std=0.3081, pad=4)


def create_layer(
    name: str, layer_type: str, hparams: Optional[Dict[str, Any]] = None
) -> tf.keras.layers.Layer:
    if layer_type == "input":
        return tf.keras.layers.Input(name=name, **hparams)

    if layer_type == "dense":
        return tf.keras.layers.Dense(name=name, **hparams)

    if layer_type == "conv2d":
        return tf.keras.layers.Conv2D(name=name, **hparams)
    
    if layer_type == "inst_norm":
        return tfa.layers.InstanceNormalization()

    if layer_type == "activation":
        return tf.keras.layers.Activation(name=name, **hparams)

    if layer_type == "dropout":
        return tf.keras.layers.Dropout(name=name, **hparams)

    if layer_type == "flatten":
        return tf.keras.layers.Flatten(name=name)

    if layer_type == "primary_capsule":
        return PrimaryCapsule(name=name, **hparams)

    if layer_type == "capsule_dr":
        return CapsuleDR(name=name, **hparams)

    if layer_type == "capsule_sa":
        return CapsuleSA(name=name, **hparams)

    if layer_type == "capsule_length":
        return CapsuleLength(name=name, **hparams)

    if layer_type == "capsule_mask":
        return CapsuleMask(name=name, **hparams)


def create_optimizer(
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
    kwargs: Union[Dict[str, Any], str],
) -> Union[str, tf.keras.optimizers.Optimizer]:
    if isinstance(kwargs, str):
        return kwargs

    kwargs["hparams"]["weight_decay"] = float(kwargs["hparams"]["weight_decay"])

    if kwargs["kind"] == "sgd":
        return tfa.optimizers.SGDW(
            learning_rate=learning_rate, momentum=0.9, **kwargs["hparams"]
        )

    if kwargs["kind"] == "adam":
        return tfa.optimizers.AdamW(learning_rate=learning_rate, **kwargs["hparams"])

    raise ValueError("Optimizer is not specificed correctly!")


def create_transform(dataset: str, transforms: Optional[List[str]] = None):
    dataset = dataset.upper()
    if dataset not in ["MNIST", "CIFAR10"]:
        raise ValueError("Dataset must either be mnist or cifar10")

    if dataset == "CIFAR10":
        aug_args = CIFAR10
    else:
        aug_args = MNIST

    normalize = A.Normalize(aug_args["mean"], aug_args["std"])

    if transforms is None:
        return normalize

    transforms_fn = []
    for transform in transforms:
        transform = transform.lower()

        if transform == "random_crop":
            transforms_fn.append(
                A.PadIfNeeded(
                    aug_args["size"][0] + aug_args["pad"],
                    aug_args["size"][1] + aug_args["pad"],
                    border_mode=0,
                )
            )
            transforms_fn.append(A.RandomCrop(aug_args["size"][0], aug_args["size"][1]))

        if transform == "color_jitter":
            transforms_fn.append(A.ColorJitter())

        if transform == "color_jitter_grayscale":
            transforms_fn.append(A.OneOf([A.ColorJitter(p=1.0), A.ToGray(p=1.0)]))

        if transform == "horizontal_flip":
            transforms_fn.append(A.HorizontalFlip())

        if transform == "vertical_flip":
            transforms_fn.append(A.VerticalFlip())

        if transform == "cutout":
            transforms_fn.append(A.CoarseDropout())

    transforms_fn.append(normalize)
    return A.Compose(transforms_fn)
