"""Main file for training the model."""
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
    loss_function_softmax = keras.losses.Huber(reduction='sum_over_batch')

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4)

    keras_model.compile(optimizer=optimizer, loss={"output_1": loss_function_label, "output_2": loss_function_softmax}, loss_weights={"output_1": 1, "output_2": 3}, metrics={'output_1': [keras.metrics.SparseCategoricalAccuracy(name='Top-1 Accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='Top-5 Accuracy')]})

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