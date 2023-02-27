from typing import Optional

import tensorflow as tf
from einops import einsum

from .activation import Squash


class Router(tf.keras.layers.Layer):
    def __init__(
            self,
            squash: Optional[str],
            use_bias: Optional[bool] = True,
            bias_initializer: Optional[str] = "zeros",
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.squash = Squash(squash) if squash is not None else tf.identity
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

    def build(self, input_shape: tf.TensorShape) -> None:
        output_capsules, output_units = input_shape[2:]
        self.bias = self.add_weight(
            "bias",
            (output_capsules, output_units),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=self.use_bias,
            constraint=self.bias_constraint,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = inputs + self.bias
        outputs = self.squash(outputs)
        return outputs


class DynamicRouter(Router):
    def __init__(
            self,
            num_routing: Optional[int] = 3,
            squash: Optional[str] = "dynamic_routing",
            use_bias: Optional[bool] = True,
            bias_initializer: Optional[str] = "zeros",
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(
            squash,
            use_bias,
            bias_initializer,
            bias_regularizer,
            bias_constraint,
            **kwargs
        )
        if num_routing < 1:
            raise ValueError("num_routing must be greater than one!")
        self.num_routing = num_routing

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        priors = tf.zeros_like(inputs)
        for i in range(self.num_routing):
            agreements = tf.nn.softmax(priors, axis=2)
            outputs = tf.reduce_sum(inputs * agreements, axis=1, keepdims=True)
            outputs = super().call(outputs)
            if i < self.num_routing - 1:
                updates = tf.reduce_sum(inputs * outputs, axis=-1, keepdims=True)
                priors = priors + updates
        return outputs[:, 0]


class SelfAttentionRouter(Router):
    def __init__(
            self,
            squash: Optional[str] = "self_attention",
            use_bias: Optional[bool] = True,
            bias_initializer: Optional[str] = "zeros",
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(
            squash,
            use_bias,
            bias_initializer,
            bias_regularizer,
            bias_constraint,
            **kwargs
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        input_capsules, output_capsules = input_shape[1:-1]
        self.bias = self.add_weight(
            "bias",
            (input_capsules, output_capsules, 1),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=self.use_bias,
            constraint=self.bias_constraint,
        )
        self.pattern = "... n_in1 n_out d_out, ... n_in2 n_out d_out -> ... n_in1 n_out"

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        vector_size = inputs.shape[-1]
        outputs = einsum(inputs, inputs, self.pattern)
        outputs = tf.expand_dims(outputs, axis=-1)
        outputs = outputs / vector_size ** 0.5
        outputs = tf.nn.softmax(outputs, axis=2)
        outputs = outputs + self.bias
        outputs = tf.reduce_sum(inputs * outputs, axis=1)
        outputs = self.squash(outputs)
        return outputs
