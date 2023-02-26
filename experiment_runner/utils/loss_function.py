from typing import Optional

import tensorflow as tf


class Margin(tf.keras.losses.Loss):
    def __init__(
            self,
            pos_weight: Optional[float] = 0.9,
            lambda_: Optional[float] = 0.5,
            reduction: Optional[tf.keras.losses.Reduction] = tf.keras.losses.Reduction.AUTO,
            name: Optional[str] = None,
    ) -> None:
        super().__init__(reduction, name)
        self.pos_weight = pos_weight
        self.lambda_ = lambda_

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss_pos = y_true * tf.nn.relu(self.pos_weight - y_pred)
        loss_neg = (1 - y_true) * tf.nn.relu(y_pred + self.pos_weight - 1)

        loss = loss_pos ** 2 + self.lambda_ * loss_neg ** 2
        loss = tf.reduce_sum(loss, -1)

        return loss
