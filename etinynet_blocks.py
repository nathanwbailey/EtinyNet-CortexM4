"""Building Blocks for EtinyNet."""
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module

class DenseLinearBottleneckBlock(keras.layers.Layer):
    """Custom Dense Linear Bottleneck Layer from EtinyNet."""
    def __init__(self, out_channels: int, kernel_size: int, padding: str = 'same', strides: int = 1, downsample: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.depthwise_conv_layer_a = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, strides=strides, use_bias=bias)
        self.depthwise_a_batch_norm_layer = keras.layers.BatchNormalization()

        self.pointwise_layer = keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', strides=1, use_bias=bias)
        self.pointwise_batch_norm = keras.layers.BatchNormalization()

        self.depthwise_conv_layer_b = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", strides=1, use_bias=bias)
        self.depthwise_b_batch_norm_layer = keras.layers.BatchNormalization()

        self.activation = keras.layers.Activation('relu')
        self.downsample_layers = None
        if downsample:
            self.downsample_layers = keras.Sequential(
                [
                    keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', strides=strides, use_bias=True),
                    keras.layers.BatchNormalization()
                ]
            )


    def call(self, input_tensor: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Forward Pass for the Dense Linear Bottleneck Layer."""
        residual = input_tensor
        depthwise_a_result = self.depthwise_a_batch_norm_layer(self.depthwise_conv_layer_a(input_tensor), training=training)
        pointwise_result = self.activation(self.pointwise_batch_norm(self.pointwise_layer(depthwise_a_result), training=training))
        depthwise_b_result = self.depthwise_b_batch_norm_layer(self.depthwise_conv_layer_b(pointwise_result), training=training)
        if self.downsample_layers:
            residual = self.downsample_layers(input_tensor, training=training)
        output = self.activation(residual + depthwise_b_result)
        return output

class LinearBottleneckBlock(keras.layers.Layer):
    """Custom Linear Bottleneck Layer from EtinyNet."""
    def __init__(self, out_channels: int, kernel_size: int, padding: str = 'same', strides: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.depthwise_conv_layer_a = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, strides=strides, use_bias=bias)
        self.depthwise_a_batch_norm_layer = keras.layers.BatchNormalization()

        self.pointwise_layer = keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', strides=1, use_bias=bias)
        self.pointwise_batch_norm = keras.layers.BatchNormalization()

        self.depthwise_conv_layer_b = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", strides=1, use_bias=bias)
        self.depthwise_b_batch_norm_layer = keras.layers.BatchNormalization()

        self.activation = keras.layers.Activation('relu')

    def call(self, input_tensor: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Forward Pass for the Linear Bottleneck Layer."""
        depthwise_result = self.depthwise_a_batch_norm_layer(self.depthwise_conv_layer_a(input_tensor), training=training)
        pointwise_result = self.activation(self.pointwise_batch_norm(self.pointwise_layer(depthwise_result), training=training))
        output = self.activation(self.depthwise_b_batch_norm_layer(self.depthwise_conv_layer_b(pointwise_result), training=training))
        return output