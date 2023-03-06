from typing import Optional

import tensorflow as tf

__all__ = ["DenseBlock"]


class DenseBlock(tf.keras.layers.Layer):
    def __init__(
        self, filters: int, blocks: int, activation: Optional[str] = "relu", **kwargs
    ):
        super().__init__(**kwargs)
        self.convs = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(activation),
                    tf.keras.layers.Conv2D(4 * i * filters, 1),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(activation),
                    tf.keras.layers.Conv2D(i * filters, 3, padding="same"),
                ]
            )
            for i in range(1, blocks + 1)
        ]

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        outputs = self.convs[0](inputs, training)
        for layer in self.convs[1:-1]:
            outputs = tf.concat([outputs, layer(outputs, training)], axis=-1)
        return self.convs[-1](outputs)
