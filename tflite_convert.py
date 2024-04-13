"""Main file for training/converting the model."""
from typing import Generator
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
import os

# --------------------------------------- #
# -- Create the Dataset from Directory -- #
# --------------------------------------- #

BATCH_SIZE=128

train_dataset = keras.preprocessing.image_dataset_from_directory(
    'tiny-imagenet-200/train',
    labels='inferred',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(48,48),
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
    image_size=(48,48),
    interpolation="bilinear",
    shuffle=False,
)

print(f'Number of Validation Classes: {len(valid_dataset.class_names)}')

augment_layer = keras.layers.RandomFlip('horizontal')
norm_layer = keras.layers.Normalization()
norm_layer.adapt(rescaled_train_dataset_data.rebatch(1).take(100))

train_dataset = train_dataset.map(lambda x, y: (augment_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (rescale_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (norm_layer(x), y))

valid_dataset = valid_dataset.map(lambda x, y: (rescale_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (norm_layer(x), y))

normalized_train_dataset_data = rescaled_train_dataset_data.map(lambda x: norm_layer(x)) # pylint: disable=unnecessary-lambda


# ----------------------- #
# -- Convert to TFLite -- #
# ----------------------- #
#FP32 Model
converter = tf.lite.TFLiteConverter.from_saved_model('etinynet')
tflite_model = converter.convert()
with open("etinynet.tflite", "wb") as f:
    f.write(tflite_model) # type: ignore[reportAttributeAccessIssue]

tflite_model_kb_size = os.path.getsize("etinynet.tflite") / 1024
print(tflite_model_kb_size)

#INT8 Model
def representative_dataset_function() -> Generator[list, None, None]:
    """Create a representative dataset for TFLite Conversion."""
    for input_value in normalized_train_dataset_data.rebatch(1).take(1000):
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