from typing import Any, Dict, Optional, Union

import tensorflow as tf
from utils.layer import *
import tensorflow_addons as tfa


def generate_layer(
        name: str, layer_type: str, hparams: Optional[Dict[str, Any]] = None
) -> tf.keras.layers.Layer:
    if layer_type == "input":
        return tf.keras.layers.Input(name=name, **hparams)

    if layer_type == "dense":
        return tf.keras.layers.Dense(name=name, **hparams)

    if layer_type == "conv2d":
        return tf.keras.layers.Conv2D(name=name, **hparams)

    if layer_type == "activation":
        return tf.keras.layers.Activation(name=name, **hparams)

    if layer_type == "dropout":
        return tf.keras.layers.Dropout(name=name, **hparams)

    if layer_type == "flatten":
        return tf.keras.layers.Flatten(name=name)
    
    if layer_type == "primarycaps":
        return PrimaryCapsule(name=name,**hparams)
    
    if layer_type == "CapsuleDR":
        return CapsuleDR(name=name,**hparams)
    
    if layer_type == "CapsuleSA":
        return CapsuleSA(name=name,**hparams)
    
    if layer_type == "CapsuleLength":
        return CapsuleSA(name=name,**hparams)
    
    if layer_type == "CapsuleMask":
        return CapsuleSA(name=name,**hparams)


def create_optimizer(
        lr,kwargs: Union[Dict[str, Any], str]
) -> Union[str, tf.keras.optimizers.Optimizer]:
    if isinstance(kwargs, str):
        return kwargs

    #kwargs["hparams"]["learning_rate"] = float(kwargs["hparams"]["learning_rate"])
    kwargs["hparams"]["weight_decay"] = float(kwargs["hparams"]["weight_decay"])

    if kwargs["kind"] == "sgd":
        return tfa.optimizers.SGDW(learning_rate = lr,momentum=0.9,**kwargs["hparams"])

    if kwargs["kind"] == "adam":
        return tfa.optimizers.AdamW(learning_rate = lr,**kwargs["hparams"])


def create_loss_function(kwargs: Union[Dict[str, Any], str]) -> tf.keras.losses.Loss:
    if isinstance(kwargs, str):
        return kwargs

    if kwargs["kind"] == "sparse_cross_entropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(**kwargs["hparams"])

    if kwargs["kind"] == "cross_entropy":
        return tf.keras.losses.CategoricalCrossentropy(**kwargs["hparams"])

    if kwargs["kind"] == "binary_cross_entropy":
        return tf.keras.losses.BinaryCrossentropy(**kwargs["hparams"])

    raise ValueError("Loss function is not specified correctly!")
