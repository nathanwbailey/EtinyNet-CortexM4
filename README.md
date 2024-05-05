# EtinyNet-0.75 on Arduino Nano 33 BLE Sense Rev 2

### What is this project?

This project implements the EtinyNet-0.75 CNN (https://ojs.aaai.org/index.php/AAAI/article/view/20387) for Tiny ImageNet-200. 

This is implemented in Keras and then converted to TFLite. Then it is deployed using TfLite-Micro on an Arduino Nano 33 BLE Sense Rev 2 which has a Cortex-M4F Microcontroller.

This branch trains a EtinyNet model using a student-teacher method and slowly steps down the input size of the training data from 224x224 to 48x48

### Blogs

In addition to the code, I wrote two blogs on this project that can be found here:

https://nathanbaileyw.medium.com/finding-the-limits-of-tinyml-deploying-etinynet-on-a-cortex-m4-32b3a4d21414
https://nathanbaileyw.medium.com/deploying-etinynet-on-a-cortex-m4-student-teacher-methods-957c4be7825f

### Where is the code?

* main.py - Implements EtinyNet in Keras. Steps down the input size of the training data using a student-teacher like method
* dataset_student_teacher.py - Logic for the custom student-teacher Keras dataset
* generate_xml_file.py - Creates an xml file to be used in the dataset (contains image file names + labels)
* create_student_teacher_data.py - Runs the data through the network trained on the 224x224 input images and outputs the results to an xml file used in the dataset
temperature_softmax_activation_layer.py - Custom temperature softmax activation layer used in the EtinyNet network


### Requirements

All pip packages needed can be found in requirements.txt

### How to Run

1. Download the dataset: http://cs231n.stanford.edu/tiny-imagenet-200.zip
2. Extract the dataset and place in current working directory
3. Run the python file: python3 main.py
