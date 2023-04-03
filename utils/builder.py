from typing import Any, Union

import albumentations as A
import tensorflow_addons as tfa
from keras_cv import models

from .model import *

CIFAR10 = dict(
    size=(32, 32), mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), pad=8
)

MNIST = dict(size=(28, 28), mean=0.1307, std=0.3081, pad=4)


def create_layer(
        name: str, layer_type: str, hparams: Optional[Dict[str, Any]] = None
) -> tf.keras.layers.Layer:
    if layer_type == "dense":
        return tf.keras.layers.Dense(name=name, **hparams)

    if layer_type == "conv2d":
        return tf.keras.layers.Conv2D(name=name, **hparams)

    if layer_type == "batch_norm":
        return tf.keras.layers.BatchNormalization(name=name, **hparams)

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

    if layer_type == 'dense_block':
        return DenseBlock(name=name, **hparams)


def create_optimizer(
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
        kwargs: Union[Dict[str, Any], str],
) -> Union[str, tf.keras.optimizers.Optimizer]:
    if isinstance(kwargs, str):
        return kwargs

    kwargs["hparams"]["weight_decay"] = float(kwargs["hparams"]["weight_decay"])

    if kwargs["kind"] == "sgd":
        return tfa.optimizers.SGDW(
            learning_rate=learning_rate, **kwargs["hparams"]
        )

    if kwargs["kind"] == "adam":
        return tfa.optimizers.AdamW(learning_rate=learning_rate, **kwargs["hparams"])

    raise ValueError("Optimizer is not specified correctly!")


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


def create_model(model_desc: Dict[str, Any], input_size: Tuple[int], num_classes: int) -> tf.keras.Model:
    if model_desc["kind"] in ["standard", "capsule_network"]:
        pretrained_args = dict(
            include_rescaling=False,
            include_top=True,
            input_shape=input_size,
            classes=num_classes,
            classifier_activation="linear",
        )
        if model_desc["name"] == "resnet18":
            backbone = models.resnet_v2.ResNet18V2(**pretrained_args)
        elif model_desc["name"] == "resnet34":
            backbone = models.resnet_v2.ResNet34V2(**pretrained_args)
        elif model_desc["name"] == "resnet50":
            backbone = models.resnet_v2.ResNet50V2(**pretrained_args)
        elif model_desc["name"] == "densenet121":
            backbone = models.densenet.DenseNet121(**pretrained_args)
        elif model_desc["name"] == "densenet169":
            backbone = models.densenet.DenseNet169(**pretrained_args)
        elif model_desc["name"] == "vit-tiny":
            backbone = models.vit.ViTTiny16(**pretrained_args)
        elif model_desc["name"] == "vit-small":
            backbone = models.vit.ViTS16(**pretrained_args)
        elif model_desc["name"] == "vit-base":
            backbone = models.vit.ViTB16(**pretrained_args)
        else:
            backbone = tf.keras.Sequential(name=model_desc["name"])
            backbone.add(tf.keras.layers.Input(input_size))
            for name, options in model_desc["layer"].items():
                if "hparams" not in options:
                    options["hparams"] = {}
                backbone.add(
                    create_layer(
                        name, options["layer_type"], options["hparams"]
                    )
                )

        if model_desc["kind"] == "standard":
            model = StandardModel(backbone)
        else:
            model = CapsNet(backbone)

    else:
        encoder = tf.keras.Sequential(name=model_desc["encoder"]["name"])
        decoder = tf.keras.Sequential(name=model_desc["decoder"]["name"])

        encoder.add(tf.keras.layers.Input(input_size))
        for name, options in model_desc["encoder"]["layer"].items():
            if "hparams" not in options:
                options["hparams"] = {}
            encoder.add(
                create_layer(name, options["layer_type"], options["hparams"])
            )

        for name, options in model_desc["decoder"]["layer"].items():
            if "hparams" not in options:
                options["hparams"] = {}
            decoder.add(
                create_layer(name, options["layer_type"], options["hparams"])
            )

        model = CapsNetWithDecoder(encoder, decoder)

    return model
