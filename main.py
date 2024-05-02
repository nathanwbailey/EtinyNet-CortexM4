"""Main file for training/converting the model."""
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from etinynet import create_etinynet_model
from typing import Generator
import numpy as np
import os
from dataset_student_teacher import TinyImageNetDataset
import time
from pathlib import Path


def create_dataset(batch_size: int, image_size: tuple[int]) -> tuple[keras.utils.Sequence, keras.utils.Sequence]:
    """Create the train and valid datasets, normalize and scale them."""
    transforms = keras.Sequential([
        keras.layers.Resizing(*image_size),
        keras.layers.Rescaling(1./255)
    ])

    train_dataset = TinyImageNetDataset(dataset_type='train', dataset_path='tiny-imagenet-200', batch_size=batch_size, transforms=transforms)

    mean = tf.zeros(3)
    variance = tf.zeros(3)

    for data, _ in train_dataset:
        image_mean = tf.math.reduce_mean(data, axis=(0,1,2))
        image_variance = tf.math.reduce_variance(data, axis=(0,1,2))
        mean = tf.math.add(mean, image_mean)
        variance = tf.math.add(variance, image_variance)

    mean = (mean/len(train_dataset))
    variance = (variance/len(train_dataset))

    train_transforms = keras.Sequential([
        keras.layers.Resizing(*image_size),
        keras.layers.RandomFlip('horizontal'),
        keras.layers.Rescaling(1./255),
        keras.layers.Normalization(mean=mean, variance=variance)
    ])

    val_transforms = keras.Sequential([
        keras.layers.Resizing(*image_size),
        keras.layers.Rescaling(1./255),
        keras.layers.Normalization(mean=mean, variance=variance)
    ])

    train_transforms_normalized = keras.Sequential([
        keras.layers.Resizing(*image_size),
        keras.layers.Rescaling(1./255),
        keras.layers.Normalization(mean=mean, variance=variance)
    ])

    train_dataset = TinyImageNetDataset(dataset_type='train', dataset_path='tiny-imagenet-200', batch_size=batch_size, transforms=train_transforms)
    train_dataset_normalized = TinyImageNetDataset(dataset_type='train', dataset_path='tiny-imagenet-200', batch_size=batch_size, transforms=train_transforms_normalized)
    val_dataset = TinyImageNetDataset(dataset_type='val', dataset_path='tiny-imagenet-200', batch_size=batch_size, transforms=val_transforms)
    return train_dataset, val_dataset, train_dataset_normalized

# class PrintOutputsCallback(keras.callbacks.Callback):
#     """Keras callback to print outputs and labels for manual comparison."""
#     def __init__(self, dataset):
#         super().__init__()
#         self.dataset = dataset

#     def on_batch_end(self, batch, logs=None):
#         batch_x, labels = self.dataset[batch]
#         _ , output2 = self.model.predict_on_batch(batch_x)
#         print(f"Model output after batch {batch}: {labels[1][0]}, {output2[0]}")


def train_model(keras_model: keras.Model, learning_rate: float, t_dataset: keras.utils.Sequence, v_dataset: keras.utils.Sequence, epochs: int) -> keras.Model:
    """Compile and train the model."""
    loss_function_label = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_function_softmax = keras.losses.Huber(reduction='sum')

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4)

    keras_model.compile(optimizer=optimizer, loss={"output_1": loss_function_label, "output_2": loss_function_softmax}, loss_weights={"output_1": 1, "output_2": 3}, metrics={'output_1': [keras.metrics.SparseCategoricalAccuracy(name='Top-1 Accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='Top-5 Accuracy')]}

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7, min_delta=1e-4)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10)

    keras_model.fit(
        t_dataset,
        epochs=epochs,
        validation_data=v_dataset,
        # callbacks = [lr_scheduler, early_stopping, PrintOutputsCallback(t_dataset)],
        callbacks = [lr_scheduler, early_stopping],
        verbose=2
    )
    return keras_model


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
        "block_type": "lb",
        "layer_values": [{"out_channels": 168} for _ in range(3)]
    },
    {
        "block_type": "lb",
        "layer_values": [{"out_channels": 192} for _ in range(1)] + [{"out_channels": 384}]
    }
]


# -------------------- #
# -- Model Training -- #
# -------------------- #

BATCH_SIZE=128

def train_model_input_size(spatial_image_size: tuple[int], input_model: keras.Model | None = None) -> keras.Model:
    """Train a model with different input size after reading in previous weights."""
    train_dataset, valid_dataset, _  = create_dataset(batch_size=BATCH_SIZE, image_size=spatial_image_size)
    i_shape = tuple(list(spatial_image_size) + [3])

    new_model = create_etinynet_model(i_shape, block_info=etinynet_block_info, initial_in_channels=24, output_units=200)
    if input_model:
        new_model.set_weights(input_model.get_weights())
    new_model.summary(expand_nested=True)

    new_model = train_model(keras_model=new_model, learning_rate=0.1, t_dataset=train_dataset, v_dataset=valid_dataset, epochs=100)
    new_model.save('etinynet_'+str(spatial_image_size[0]))
    return new_model

input_tuples = [(112, 112), (96, 96), (64, 64), (48, 48)]

model = keras.models.load_model('etinynet_224_student_teacher_baseline')
for idx, input_tuple in enumerate(input_tuples):
    model = train_model_input_size(input_tuple, input_model=model)


final_image_size = (48,48)
train_dataset_final, valid_dataset_final, train_dataset_normalized = create_dataset(batch_size=BATCH_SIZE, image_size=final_image_size)

# ----------------------- #
# -- Convert to TFLite -- #
# ----------------------- #
#FP32 Model
converter = tf.lite.TFLiteConverter.from_saved_model('etinynet_48')
tflite_model = converter.convert()
with open("etinynet.tflite", "wb") as f:
    f.write(tflite_model) # type: ignore[reportAttributeAccessIssue]

tflite_model_kb_size = os.path.getsize("etinynet.tflite") / 1024
print(tflite_model_kb_size)

#INT8 Model
def representative_dataset_function() -> Generator[list, None, None]:
    """Create a representative dataset for TFLite Conversion."""
    for input_value, _ in train_dataset_normalized:
        i_value_fp32 = tf.cast(input_value, tf.float32)
        yield [i_value_fp32]

converter = tf.lite.TFLiteConverter.from_saved_model('etinynet_48')
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
    input_data = tf.reshape(input_data, (1,48,48,3))
    input_fp32 = tf.cast(input_data, tf.float32)
    input_int8 = tf.cast((input_fp32 / i_scale) + i_zero_point, tf.int8)
    interpreter.set_tensor(input_d["index"], input_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_d["index"])[0]
    output_fp32 = tf.convert_to_tensor((output_int8 - o_zero_point) * o_scale, dtype=tf.float32)
    return output_fp32

num_correct_examples = 0
num_examples = 0
num_correct_examples_top_5 = 0
for i_value, o_value in valid_dataset_final:
    output = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)
    if tf.cast(tf.math.argmax(output), tf.int32) == o_value:
        num_correct_examples += 1
    if tf.math.in_top_k(tf.expand_dims(o_value, axis=0), tf.expand_dims(output, axis=0), 5).numpy()[0]:
        num_correct_examples_top_5 += 1
    num_examples += 1

print(f'Top-1 Accuracy: {num_correct_examples/num_examples}')
print(f'Top-5 Accuracy: {num_correct_examples_top_5/num_examples}')

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


# filtered_valid_dataset = valid_dataset_final.filter(lambda _, y: y == 115)

# c_code = ""
# for i_value, o_value in filtered_valid_dataset:
#     o_pred_fp32 = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)
#     if tf.cast(tf.math.argmax(o_pred_fp32), tf.int32) == o_value:
#         i_value_int8 = tf.cast(((i_value / input_quant_scale) + input_quant_zero_point), tf.int8).numpy()
#         i_value_int8 = i_value_int8.ravel()
#         val_str = array_to_str(i_value_int8)
#         c_code = generate_h_file(i_value_int8.size, val_str, "115")

# with open('input_imagenet.h', 'w', encoding='utf-8') as file:
#     file.write(c_code)