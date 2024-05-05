"""Add the teacher's softmaxed output to the labels in the dataset xml."""
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from dataset import TinyImageNetDataset
import xml.etree.ElementTree as ET

def temperature_softmax(logits, temperature=1.0):
    """Softmax activation function with temperature."""
    return keras.activations.softmax(logits / temperature, axis=1)

mean = tf.constant([0.4802367, 0.44806668, 0.3975034])
variance = tf.constant([0.06806142, 0.06479827, 0.06956852])

train_transforms = keras.Sequential([
    keras.layers.Resizing(224,224),
    keras.layers.RandomFlip('horizontal'),
    keras.layers.Rescaling(1./255),
    keras.layers.Normalization(mean=mean, variance=variance)
])


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

model = keras.models.load_model('etinynet_224_student_teacher_baseline')

mytree = ET.parse('tiny-imagenet-200/train/image_labels.xml')
root = mytree.getroot()

for idx, (data, labels) in enumerate(train_dataset):
    model_output = model.predict(data, verbose=0)
    softmax_output = temperature_softmax(tf.convert_to_tensor(model_output), temperature=2)
    numpy_softmax_output = softmax_output.numpy()
    for i in range(data.shape[0]):
        root[0][i+idx*32].attrib['label'] = str(list(numpy_softmax_output[i]))

tree = ET.ElementTree(root)
tree.write("tiny-imagenet-200/train/image_labels_teacher_student.xml")

mytree = ET.parse('tiny-imagenet-200/val/image_labels.xml')
root = mytree.getroot()

for idx, (data, labels) in enumerate(val_dataset):
    model_output = model.predict(data, verbose=0)
    softmax_output = temperature_softmax(tf.convert_to_tensor(model_output), temperature=2)
    numpy_softmax_output = softmax_output.numpy()
    for i in range(data.shape[0]):
        root[0][i+idx*32].attrib['label'] = str(list(numpy_softmax_output[i]))

tree = ET.ElementTree(root)
tree.write("tiny-imagenet-200/val/image_labels_teacher_student.xml")