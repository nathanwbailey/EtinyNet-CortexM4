"""Tempature Softmax Activation Layer."""
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from typing import Any

class TempatureSoftmaxActivationLayer(keras.layers.Layer):
    """Custom Tempature Softmax Activation Layer."""
    def __init__(self, tempature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tempature = tempature

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        return keras.activations.softmax(input_tensor / self.tempature, axis=1)