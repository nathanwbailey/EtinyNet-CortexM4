"""Main file for training/converting the model."""
from typing import cast
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
import numpy as np

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
norm_layer.adapt(rescaled_train_dataset_data.rebatch(1).take(100))

train_dataset = train_dataset.map(lambda x, y: (augment_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (rescale_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (norm_layer(x), y))

valid_dataset = valid_dataset.map(lambda x, y: (rescale_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (norm_layer(x), y))

normalized_train_dataset_data = rescaled_train_dataset_data.map(lambda x: norm_layer(x)) # pylint: disable=unnecessary-lambda

# --------------------------- #
# -- Evaluate TFLite Model -- #
# --------------------------- #

tflite_interpreter = tf.lite.Interpreter(model_path="etinynet_int8.tflite")
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()[0]
output_details = tflite_interpreter.get_output_details()[0]

input_quantization_details = input_details["quantization_parameters"]
output_quantization_details = output_details["quantization_parameters"]
input_quant_scale = input_quantization_details['scales'][0]
output_quant_scale = output_quantization_details['scales'][0]
input_quant_zero_point = input_quantization_details['zero_points'][0]
output_quant_zero_point = output_quantization_details['zero_points'][0]

def classify_sample_tflite(interpreter: tf.lite.Interpreter, input_d: dict, output_d: dict, i_scale: np.float32, o_scale: np.float32, i_zero_point: np.int32, o_zero_point: np.int32, input_data: tf.Tensor) -> tf.Tensor:
    """Classify an example in TFLite."""
    input_data = tf.reshape(input_data, (1,224,224,3))
    input_fp32 = tf.cast(input_data, tf.float32)
    input_int8 = tf.cast(((input_fp32 / i_scale) + i_zero_point), tf.int8)
    interpreter.set_tensor(input_d["index"], input_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_d["index"])[0]
    output_fp32 = tf.convert_to_tensor((output_int8 - o_zero_point) * o_scale, dtype=tf.float32)
    return output_fp32

num_correct_examples = 0
num_examples = 0
num_correct_examples_top_5 = 0
for i_value, o_value in valid_dataset.unbatch():
    output = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)
    if tf.cast(tf.math.argmax(output), tf.int32) == o_value:
        num_correct_examples += 1
    if tf.math.in_top_k(tf.expand_dims(o_value, axis=0), tf.expand_dims(output, axis=0), 5).numpy()[0]:
        num_correct_examples_top_5 += 1
    num_examples += 1

print(f'Top-1 Accuracy: {num_correct_examples/num_examples}')
print(f'Top-5 Accuracy: {num_correct_examples_top_5/num_examples}')
