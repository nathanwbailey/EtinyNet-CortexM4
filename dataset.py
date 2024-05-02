import numpy as np
import xml.etree.ElementTree as ET 
import math
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
from typing import Literal
from typing import Any
from skimage.io import imread
from PIL import Image
from sklearn.utils import shuffle


class TinyImageNetDataset(keras.utils.Sequence):
    """Dataset class for Tiny Image Net."""
    def __init__(self, dataset_type: Literal['train', 'val'], dataset_path: str, batch_size: int, transforms: keras.Sequential | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        xml_file = open(dataset_path+'/'+dataset_type+'/image_labels.xml', mode='r', encoding='UTF-8')
        xml_root = ET.fromstring(xml_file.read())
        self.batch_size = batch_size
        self.names = []
        self.labels = []
        for item in xml_root.findall('Items/Item'):
            self.names.append(dataset_path+'/'+dataset_type+'/'+item.attrib['imageName'])
            self.labels.append(int(item.attrib['label']))
        self.transforms = transforms
        self.names, self.labels = shuffle(self.names, self.labels, random_state=20)

    def __len__(self):
        return math.ceil(len(self.names) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.names))
        names = self.names[low:high]
        labels = self.labels[low:high]

        data_batch = []
        for file_name in names:
            data_batch.append(np.asarray(Image.open(file_name).convert('RGB')))

        data_batch = np.array(data_batch)
        label_batch = np.array(labels)
        if self.transforms:
            return self.transforms(data_batch), label_batch
        return data_batch, label_batch