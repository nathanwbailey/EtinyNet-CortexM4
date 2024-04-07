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
norm_layer.adapt(rescaled_train_dataset_data.rebatch(1).take(100))

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
def representative_dataset_function() -> Generator[list, None, None]:
    """Create a representative dataset for TFLite Conversion."""
    for input_value in normalized_train_dataset_data.rebatch(1).take(100):
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


# --------------------------- #
# -- Evaluate TFLite Model -- #
# --------------------------- #
tflite_interpreter = tf.lite.Interpreter(model_content = tflite_model)
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
    input_int8 = tf.cast((input_fp32 / i_scale) + i_zero_point, tf.int8)
    interpreter.set_tensor(input_d["index"], input_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_d["index"])[0]
    output_fp32 = cast(tf.Tensor, tf.cast((output_int8 - o_zero_point) * o_scale, tf.float32))
    return output_fp32

num_correct_examples = 0
num_examples = 0
for i_value, o_value in valid_dataset.unbatch():
    output = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)
    if tf.cast(tf.math.argmax(output), tf.int32) == o_value:
        num_correct_examples += 1
    num_examples += 1

print(f'Accuracy: {num_correct_examples/num_examples}')


# ------------------------- #
# -- Prepare for Arduino -- #
# ------------------------- #
def array_to_str(data: np.ndarray) -> str:
    """Convert numpy array of int8 values to comma seperated int values."""
    num_cols = 10
    val_string = ''
    for i, val in enumerate(data):
        val_string += str(val)
        if (i+1) < len(data):
            val_string += ','
        if (i+1) % num_cols == 0:
            val_string += '\n'
    return val_string

def generate_h_file(size: int, data: str, label: str) -> str:
    """Generate a c header with the string numpy data."""
    str_out = 'int8_t g_test[] = '
    str_out += '\n{\n'
    str_out += f'{data}'
    str_out += '};\n'
    str_out += f'const int g_test_len = {size};\n'
    str_out += f'const int g_test_label = {label};\n'
    return str_out


filtered_valid_dataset = valid_dataset.unbatch().filter(lambda _, y: y == 115)

c_code = ""
for i_value, o_value in filtered_valid_dataset:
    o_pred_fp32 = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)
    if tf.cast(tf.math.argmax(output), tf.int32) == o_value:
        i_value_int8 = ((i_value / input_quant_scale) + input_quant_zero_point).astype(np.int8)
        i_value_int8 = i_value_int8.ravel()
        val_str = array_to_str(i_value_int8)
        c_code = generate_h_file(i_value_int8.size, val_str, "6")

with open('input_imagenet.h', 'w', encoding='utf-8') as file:
    file.write(c_code)
