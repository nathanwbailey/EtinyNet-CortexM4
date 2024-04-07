"""Main file for training/converting the model."""
from typing import Generator
from typing import cast
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from pathlib import Path
import numpy as np
import pandas as pd
import os
import time
from etinynet import create_etinynet_model

# --------------------------------------- #
# -- Create the Dataset from Directory -- #
# --------------------------------------- #

BATCH_SIZE=128

train_dataset = keras.preprocessing.image_dataset_from_directory(
    'tiny-imagenet-200/train',
    labels='inferred',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(224,224),
    interpolation="bilinear",
    shuffle=True,
    seed=123,
)

num_train_classes = len(train_dataset.class_names)

rescale_layer = keras.layers.Rescaling(1./255)
rescaled_train_dataset = train_dataset.map(lambda x, y: (rescale_layer(x), y))
rescaled_train_dataset_data = rescaled_train_dataset.map(lambda data, _ : data)

print(f'Number of Training Classes: {num_train_classes}')

valid_dataset = keras.preprocessing.image_dataset_from_directory(
    'tiny-imagenet-200/val',
    labels='inferred',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(224,224),
    interpolation="bilinear",
    shuffle=False,
)

print(f'Number of Validation Classes: {len(valid_dataset.class_names)}')

augment_layer = keras.layers.RandomFlip('horizontal')
norm_layer = keras.layers.Normalization()
norm_layer.adapt(rescaled_train_dataset_data)

train_dataset = train_dataset.map(lambda x, y: (augment_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (rescale_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (norm_layer(x), y))

valid_dataset = valid_dataset.map(lambda x, y: (rescale_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (norm_layer(x), y))

normalized_train_dataset_data = rescaled_train_dataset_data.map(lambda x: norm_layer(x)) # pylint: disable=unnecessary-lambda

# ------------------------------------- #
# -- Create the Classification Model -- #
# ------------------------------------- #
etinynet_block_info = [
    {
        "block_type": "lb",
        "layer_values": [{"out_channels": 24} for _ in range(4)]
    },
    {
        "block_type": "lb",
        "layer_values": [{"out_channels": 96} for _ in range(4)]
    },
    {
        "block_type": "dlb",
        "layer_values": [{"out_channels": 168} for _ in range(3)]
    },
    {
        "block_type": "dlb",
        "layer_values": [{"out_channels": 192} for _ in range(2)] + [{"out_channels": 384}]
    }
]

i_shape = (224,224,3)
model = create_etinynet_model(i_shape, block_info=etinynet_block_info, initial_in_channels=24, output_units=num_train_classes)
model.summary(expand_nested=True)

# ------------------------------------ #
# -- Train the Classification Model -- #
# ------------------------------------ #

loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
LEARNING_RATE = 0.1
optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[keras.metrics.SparseCategoricalAccuracy(name='Top-1 Accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='Top-5 Accuracy')])

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7, min_delta=1e-4)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10)

logging_directory_name = "tensorboard_log_dir"
if not Path(logging_directory_name).exists():
    Path(logging_directory_name).mkdir()
root_logdir = os.path.join(os.curdir, logging_directory_name)

def get_run_logdir(root_logdir_in: str) -> str:
    """Return a folder for the run to use in Tensorboard."""
    run_id = time.strftime("dense_run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir_in, run_id)

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir(root_logdir))

model.fit(
    train_dataset,
    epochs=1000,
    validation_data=valid_dataset,
    callbacks = [lr_scheduler, early_stopping, tensorboard_cb],
    verbose=2
)
model.save('etinynet')


# ----------------------- #
# -- Convert to TFLite -- #
# ----------------------- #
train_dataset_data = normalized_train_dataset_data.unbatch()
def representative_dataset_function() -> Generator[list, None, None]:
    """Create a representative dataset for TFLite Conversion."""
    for input_value in train_dataset_data.batch(1).take(100):
        i_value_fp32 = tf.cast(input_value, tf.float32)
        yield [i_value_fp32]

converter = tf.lite.TFLiteConverter.from_saved_model('etinynet')
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_function)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore[reportAttributeAccessIssue]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8 # type: ignore[reportAttributeAccessIssue]
converter.inference_output_type = tf.int8 # type: ignore[reportAttributeAccessIssue]


tflite_model = converter.convert()
with open("etinynet_int8.tflite", "wb") as f:
    f.write(tflite_model) # type: ignore[reportAttributeAccessIssue]

tflite_model_kb_size = os.path.getsize("etinynet_int8.tflite") / 1024
print(tflite_model_kb_size)
