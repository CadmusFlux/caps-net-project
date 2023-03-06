from typing import Optional

import tensorflow as tf

from .activation import *
from .addons import *
from .capsule import *
from .router import *


class Capsule(tf.keras.layers.Layer):
    def __init__(
            self,
            capsules: int,
            units: int,
            use_bias: Optional[bool] = True,
            kernel_initializer: Optional[str] = "glorot_uniform",
            bias_initializer: Optional[str] = "zeros",
            kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.transform = CapsuleTransform(
            capsules, units, kernel_initializer, kernel_regularizer, kernel_constraint
        )

        self.route = Router(
            "dynamic_routing",
            use_bias,
            bias_initializer,
            bias_regularizer,
            bias_constraint,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.transform(inputs)
        outputs = self.route(outputs)
        return outputs


class CapsuleDR(Capsule):
    def __init__(
            self,
            capsules: int,
            units: int,
            num_routing: Optional[int] = 3,
            use_bias: Optional[bool] = True,
            kernel_initializer: Optional[str] = "glorot_uniform",
            bias_initializer: Optional[str] = "zeros",
            kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(
            capsules,
            units,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs
        )

        self.route = DynamicRouter(
            num_routing,
            "dynamic_routing",
            use_bias,
            bias_initializer,
            bias_regularizer,
            bias_constraint,
        )


class CapsuleSA(Capsule):
    def __init__(
            self,
            capsules: int,
            units: int,
            use_bias: Optional[bool] = True,
            kernel_initializer: Optional[str] = "he_normal",
            bias_initializer: Optional[str] = "zeros",
            kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
            bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
            **kwargs
    ) -> None:
        super().__init__(
            capsules,
            units,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs
        )

        self.route = SelfAttentionRouter(
            "self_attention",
            use_bias,
            bias_initializer,
            bias_regularizer,
            bias_constraint,
        )
