import os 
from lxml import etree

def create_xml_file(path: str) -> None:
    """Create label xml file for dataset."""
    dataset_labels = os.listdir(path=path)
    dataset_labels = [label for label in dataset_labels if os.path.isdir(path+'/'+label)]
    len_dataset = 0
    for idx, label in enumerate(dataset_labels):
        list_files = os.listdir(path+'/'+label+'/images')
        len_dataset += len(list_files)

    root = etree.Element("Images")
    items = etree.SubElement(root, "Items", num_images = str(len_dataset))

    for idx, label in enumerate(dataset_labels):
        list_files = os.listdir(path+'/'+label+'/images')
        list_files.sort()
        for file in list_files:
            etree.SubElement(items, "Item", imageName=label+'/images/'+file, label=str(idx))

    tree = etree.ElementTree(root)
    tree.write(path+"/image_labels.xml", pretty_print=True)

create_xml_file(path='tiny-imagenet-200/train')
create_xml_file(path='tiny-imagenet-200/val')
