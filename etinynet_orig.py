"""EtinyNet Keras Implementation."""
from typing import Type
from typing import Union
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from etinynet_blocks import LinearBottleneckBlock
from etinynet_blocks import DenseLinearBottleneckBlock
from tempature_softmax_activation_layer import TempatureSoftmaxActivationLayer

def tempature_softmax_activation(tempature: float = 1.0):
    """Softmax activation function with tempature."""
    def tempature_softmax(logits: tf.Tensor):
        """Tempature softmax function."""
        return keras.activations.softmax(logits / tempature, axis=1)
    return tempature_softmax


#See the number of GPU devices
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def stack_linearwise_block(data: tf.Tensor, block_type: Type[Union[LinearBottleneckBlock, DenseLinearBottleneckBlock]], block_config: list[dict], initial_in_channels: int) -> tf.Tensor:
    """Stack Linear or Dense blocks together, pass through an input and return output."""

    in_channels = initial_in_channels
    for idx, config in enumerate(block_config):
        out_channels = config.get("out_channels")
        if not out_channels:
            raise KeyError('key out_channels not found in block config')
        
        padding='same'
        if idx != 0:
            stride=1
        else:
            stride=2

        extra_args = {}
        if block_type == DenseLinearBottleneckBlock:
            extra_args = {'downsample':(stride == 2 or out_channels != in_channels)}

        block = block_type(out_channels=out_channels, kernel_size=3, strides=stride, padding=padding, **extra_args)
        in_channels = out_channels
        data = block(data)

    return data


def create_stack(data: tf.Tensor, block_info: list[dict], initial_in_channels: int) -> tf.Tensor:
    """Stack blocks of blocks together, pass through an input and return output."""
    in_channels = initial_in_channels
    for block in block_info:
        block_type = block.get('block_type')
        if not block_type:
            block_type = "lb"
        
        layer_values = block.get('layer_values')

        if not layer_values:
            raise KeyError('key layer_values not found in block config')
        
        data = stack_linearwise_block(data, block_type=LinearBottleneckBlock if block_type =='lb' else DenseLinearBottleneckBlock, block_config=layer_values, initial_in_channels=in_channels)

        in_channels = layer_values[-1].get('out_channels')

        if not in_channels:
            raise KeyError('key out_channels not found in block config')
    return data



def create_etinynet_model(i_shape: tuple, block_info: list[dict], initial_in_channels: int, output_units: int) -> keras.Model:
    """Create an EtinyNet model and return it."""
    input_data = keras.layers.Input(shape=i_shape)

    out = keras.layers.Conv2D(filters=initial_in_channels, kernel_size=3, strides=2)(input_data)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = create_stack(out, block_info=block_info, initial_in_channels=initial_in_channels)

    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dropout(rate=0.4)(out)
    output_1 = keras.layers.Dense(units=output_units, name="output_1")(out)

    model = keras.Model(inputs=input_data, outputs=output_1)

    return model