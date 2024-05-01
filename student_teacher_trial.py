"""Trial file for training the model using student teacher methods."""
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from etinynet import create_etinynet_model
from typing import Generator
import numpy as np
import os
from dataset_student_teacher import TinyImageNetDataset
import time
from pathlib import Path

transforms = keras.Sequential([
    keras.layers.Resizing(224,224),
    keras.layers.Rescaling(1./255)
])

train_dataset = TinyImageNetDataset(dataset_type='train', dataset_path='tiny-imagenet-200', batch_size=32, transforms=transforms)

# mean = tf.zeros(3)
# variance = tf.zeros(3)

# for data, _, _ in train_dataset:
#     image_mean = tf.math.reduce_mean(data, axis=(0,1,2))
#     image_variance = tf.math.reduce_variance(data, axis=(0,1,2))
#     mean = tf.math.add(mean, image_mean)
#     variance = tf.math.add(variance, image_variance)

# mean = (mean/len(train_dataset))
# variance = (variance/len(train_dataset))

# print(mean)
# print(variance)

mean = tf.constant([0.48023695,0.44806597, 0.39750367])
variance = tf.constant([0.06806144, 0.06479828, 0.06956855])

train_transforms = keras.Sequential([
    keras.layers.Resizing(224,224),
    keras.layers.RandomFlip('horizontal'),
    keras.layers.Rescaling(1./255),
    keras.layers.Normalization(mean=mean, variance=variance)
])

val_transforms = keras.Sequential([
    keras.layers.Resizing(224,224),
    keras.layers.Rescaling(1./255),
    keras.layers.Normalization(mean=mean, variance=variance)
])

train_dataset = TinyImageNetDataset(dataset_type='train', dataset_path='tiny-imagenet-200', batch_size=32, transforms=train_transforms)
val_dataset = TinyImageNetDataset(dataset_type='val', dataset_path='tiny-imagenet-200', batch_size=32, transforms=val_transforms)

print(type(train_dataset))

# for data, labels in train_dataset:
#     print(data.shape)
#     print(labels[0].shape)
#     print(labels[1].shape)

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

i_shape = (224,224,3)
model = create_etinynet_model(i_shape, block_info=etinynet_block_info, initial_in_channels=24, output_units=200)
model.summary(expand_nested=True)

# ------------------------------------ #
# -- Train the Classification Model -- #
# ------------------------------------ #

loss_function_label = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function_softmax = keras.losses.Huber(reduction='sum')


LEARNING_RATE = 0.1
optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

model.compile(optimizer=optimizer, loss={"output_1": loss_function_label, "output_2": loss_function_softmax}, metrics={'output_1': [keras.metrics.SparseCategoricalAccuracy(name='Top-1 Accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='Top-5 Accuracy')]})

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7, min_delta=1e-4)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10)

model.fit(
    train_dataset,
    epochs=10000,
    validation_data=val_dataset,
    callbacks = [lr_scheduler, early_stopping]
)
model.save('etinynet')