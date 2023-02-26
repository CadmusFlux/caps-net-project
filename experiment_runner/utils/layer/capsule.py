from typing import Optional, Tuple, Union

import tensorflow as tf
from einops import einsum, rearrange

from .activation import Squash


class PrimaryCapsule(tf.keras.layers.Layer):
    def __init__(
            self,
            capsules: int,
            units: int,
            kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
            strides: Optional[Union[int, Tuple[int, int]]] = (1, 1),
            padding: Optional[str] = "valid",
            data_format: Optional[str] = None,
            dilation_rate: Optional[Union[int, Tuple[int, int]]] = (1, 1),
            depthwise: Optional[bool] = False,
            squash: Optional[str] = None,
            use_bias: Optional[bool] = False,
            kernel_initializer: Optional[str] = "glorot_uniform",
            bias_initializer: Optional[str] = "zeros",
            kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.capsules = capsules
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depthwise = depthwise
        self.squash = Squash(squash) if squash is not None else tf.identity
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape: tf.TensorShape) -> None:
        height, width, channel = input_shape[1:]

        self.pattern = "... h w (n d) -> ... (h w n) d"
        if self.data_format == "channels_first":
            channel, height, width = input_shape[1:]
            self.pattern = "... (n d) h w -> ... (h w n) d"

        if self.kernel_size is None:
            self.kernel_size = [height, width]

        self.convolve = tf.keras.layers.Conv2D(
            filters=self.capsules * self.units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            groups=channel if self.depthwise else True,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.convolve(inputs)
        outputs = rearrange(outputs, self.pattern, n=self.capsules, d=self.units)
        outputs = self.squash(outputs)
        return outputs


class CapsuleTransform(tf.keras.layers.Layer):
    def __init__(
            self,
            capsules: int,
            units: int,
            kernel_initializer: Optional[str] = "glorot_uniform",
            kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.capsules = capsules
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape: tf.TensorShape) -> None:
        input_capsules, input_units = input_shape[1:]
        self.kernel = self.add_weight(
            "kernel",
            shape=(input_capsules, input_units, self.capsules, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            constraint=self.kernel_constraint,
        )

        self.pattern = "... n_in d_in, n_in d_in n_out d_out -> ... n_in n_out d_out"

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = einsum(inputs, self.kernel, self.pattern)
        return outputs


class CapsuleLength(tf.keras.layers.Layer):
    def __init__(
            self,
            ord: Optional[Union[str, int]] = "euclidean",
            axis: Optional[int] = -1,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ord = ord
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.norm(inputs, self.ord, axis=self.axis)


class CapsuleMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor]) -> tf.Tensor:
        if mask is None:
            mask = tf.norm(tf.stop_gradient(inputs), axis=-1)
            mask = tf.argmax(mask, inputs.shape[1])
            mask = tf.one_hot(mask, inputs.shape[1])
        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        return inputs * mask
