from typing import Dict, List, Tuple

from keras.engine.data_adapter import unpack_x_y_sample_weight

from .layer import *
from .loss_function import Margin


class StandardModel(tf.keras.Sequential):
    def __init__(self, backbone: tf.keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add(backbone)
        self.build(backbone.input_shape)

    def summary(self, **kwargs) -> None:
        super().summary(**kwargs)
        self.layers[0].summary(**kwargs)

    def compile(
        self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None, **kwargs
    ) -> None:
        super().compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
            **kwargs
        )


class CapsNet(tf.keras.Sequential):
    def __init__(self, backbone: tf.keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add(backbone)
        self.add(CapsuleLength(name="classifier"))
        self.build(backbone.input_shape)

    def summary(self, **kwargs) -> None:
        super().summary(**kwargs)
        self.layers[0].summary(**kwargs)

    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        pos_weight: Optional[int] = 0.9,
        lambda_: Optional[int] = 0.5,
        **kwargs
    ) -> None:
        super().compile(
            optimizer=optimizer,
            loss=Margin(pos_weight, lambda_),
            metrics=["accuracy"],
            **kwargs
        )


class CapsNetWithDecoder(tf.keras.Model):
    def __init__(
        self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.classifier = CapsuleLength()
        self.decoder = decoder
        self.build(encoder.input_shape)

    def call(self, inputs, training=None, mask=None):
        out_encoder = self.encoder(inputs, training, mask)
        out_decoder = self.decoder(out_encoder, training, mask)
        return dict(encoder=self.classifier(out_encoder), decoder=out_decoder)

    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, float]:
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, mask=y)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        pos_weight: Optional[float] = 0.9,
        lambda_: Optional[float] = 0.5,
        rec_coef: Optional[float] = 0.005,
        **kwargs
    ) -> None:
        super().compile(optimizer=optimizer, **kwargs)
        self.loss_fn = dict(
            margin=Margin(
                pos_weight, lambda_, reduction=tf.keras.losses.Reduction.NONE
            ),
            reconstruction=tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            ),
        )

        self.rec_coef = rec_coef

        self.metrics_ = dict(
            loss=tf.keras.metrics.Mean(name="loss"),
            loss_margin=tf.keras.metrics.Mean(name="loss_margin"),
            loss_reconstruction=tf.keras.metrics.Mean(name="loss_reconstruction"),
            accuracy=tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        )

    def compute_loss(
        self,
        x: Optional[tf.Tensor] = None,
        y: Optional[tf.Tensor] = None,
        y_pred: Optional[tf.Tensor] = None,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        shape_flatten = (-1, tf.reduce_prod(self.encoder.input_shape[1:]))
        x_flatten = tf.reshape(x, shape_flatten)
        y_pred["decoder"] = tf.reshape(y_pred["decoder"], shape_flatten)

        mrg_loss = self.loss_fn["margin"](y, y_pred["encoder"])
        self.metrics_["loss_margin"].update_state(mrg_loss)

        rec_loss = self.loss_fn["reconstruction"](x_flatten, y_pred["decoder"])
        self.metrics_["loss_reconstruction"].update_state(rec_loss)

        loss = mrg_loss + self.rec_coef * rec_loss
        self.metrics_["loss"].update_state(loss)

        return tf.reduce_mean(loss)

    def compute_metrics(self, x, y, y_pred, sample_weight) -> Dict[str, float]:
        self.metrics_["accuracy"].update_state(y, y_pred["encoder"])
        return {m.name: m.result() for m in self.metrics_.values()}

    def summary(self, **kwargs) -> None:
        super().summary(**kwargs)
        self.encoder.summary(**kwargs)
        self.decoder.summary(**kwargs)

    def reset_metrics(self) -> None:
        for m in self.metrics_:
            self.metrics_[m].reset_states()

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return list(self.metrics_.values())
