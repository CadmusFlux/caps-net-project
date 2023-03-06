from typing import Optional, Literal, Union, Tuple

import tensorflow as tf

__all__ = ['Squash']


class Squash(tf.keras.layers.Layer):
    def __init__(
            self,
            squash: Literal["dynamic_routing", "self_attention"],
            ord: Optional[Union[int, str]] = "euclidean",
            axis: Optional[int] = -1,
            eps: Optional[float] = 1e-7,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if squash not in ["dynamic_routing", "self_attention"]:
            raise ValueError(f"Squash function {squash} is not defined!")
        self.squash = squash
        self.axis = axis
        self.ord = ord
        self.eps = eps

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        norm, factor = self.calculate_factor(inputs)

        return factor * inputs / (norm + self.eps)

    def calculate_factor(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        norm = tf.norm(inputs, self.ord, axis=self.axis, keepdims=True)
        if self.squash == "dynamic_routing":
            return norm, norm ** 2 / (1 + norm ** 2)
        return norm, 1 - tf.exp(-norm)
