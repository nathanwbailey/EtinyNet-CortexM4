"""EtinyNet Keras Implementation."""
from typing import Type
from typing import Union
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module

from etinynet_blocks import LinearBottleneckBlock
from etinynet_blocks import DenseLinearBottleneckBlock


#See the number of GPU devices
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def stack_linearwise_block(block_type: Type[Union[LinearBottleneckBlock, DenseLinearBottleneckBlock]], block_config: list[dict], initial_in_channels: int) -> keras.Sequential:
    """Stack blocks together and return as a Sequential Block."""
    stacked_blocks = keras.Sequential()
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

        print(extra_args)
        block = block_type(out_channels=out_channels, kernel_size=3, strides=stride, padding=padding, **extra_args)

        stacked_blocks.add(block)

        in_channels = out_channels

    return stacked_blocks


class EtinyNet(keras.Model):
    """EtinyNet Network."""
    def __init__(self, block_info: list[dict], output_units: int) -> None:
        super().__init__()
        initial_in_channels = 24
        self.initial_conv_block = keras.Sequential(
            [
                keras.layers.Conv2D(filters=initial_in_channels, kernel_size=3, strides=2),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu')
            ]
        )

        self.blocks = keras.Sequential()
        in_channels = initial_in_channels
        for block in block_info:
            block_type = block.get('block_type')
            if not block_type:
                block_type = "lb"
            
            layer_values = block.get('layer_values')

            if not layer_values:
                raise KeyError('key layer_values not found in block config')
            
            stacked_block = stack_linearwise_block(block_type=LinearBottleneckBlock if block_type =='lb' else DenseLinearBottleneckBlock, block_config=layer_values, initial_in_channels=in_channels)
            self.blocks.add(stacked_block)
            in_channels = layer_values[-1].get('out_channels')

            if not in_channels:
                raise KeyError('key out_channels not found in block config')
        self.classification_head = keras.Sequential(
            [
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(units=output_units)
            ]
        )
    
    def call(self, input_tensor: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Forward pass for EtinyNet."""
        out = self.initial_conv_block(input_tensor, training=training)
        out = self.blocks(out, training=training)
        out = self.classification_head(out, training=training)
        return out

etinynet_block_info = [
    {
        "block_type": "lb",
        "layer_values": [{"out_channels": 32} for _ in range(4)]
    },
    {
        "block_type": "lb",
        "layer_values": [{"out_channels": 128} for _ in range(4)]
    },
    {
        "block_type": "lb",
        "layer_values": [{"in_channels": 192, "out_channels": 192} for _ in range(3)]
    },
    {
        "block_type": "dlb",
        "layer_values": [{"in_channels": 192, "out_channels": 192} for _ in range(3)]
    }
]
# i_shape = (None,32,32,3)
# network = EtinyNet(block_info=etinynet_block_info, output_units=20)
# network.build(i_shape)
# print(network)
# network.summary(expand_nested=True)
# input_shape = (32, 32, 3)
# dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
# _ = network(dummy_input)