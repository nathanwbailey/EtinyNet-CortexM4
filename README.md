# EtinyNet-0.75 on Arduino Nano 33 BLE Sense Rev 2

### What is this project?

This project implements the EtinyNet-0.75 CNN (https://ojs.aaai.org/index.php/AAAI/article/view/20387) for Tiny ImageNet-200. 

This is implemented in Keras and then converted to TFLite. Then it is deployed using TfLite-Micro on an Arduino Nano 33 BLE Sense Rev 2 which has a Cortex-M4F Microcontroller.

The accuracy_increase_trial branch steps down the input size of the training data from 224x224 to 48x48 in small increments to achieve better accuracy than just training on a smaller image size from the start

The student_teacher branch steps down the input size of the training data from 224x224 to 48x48 and uses student-teacher methods to further guide the training process

### Blogs

In addition to the code, I wrote two blogs on this project that can be found here:

https://nathanbaileyw.medium.com/finding-the-limits-of-tinyml-deploying-etinynet-on-a-cortex-m4-32b3a4d21414

https://nathanbaileyw.medium.com/deploying-etinynet-on-a-cortex-m4-student-teacher-methods-957c4be7825f

### Where is the code?

* main.py - Implements EtinyNet in Keras, converts it to tflite and outputs an image in a C header.
* cortex_program/cortex_program.ino - Runs EtinyNet on the Cortex-M4F, classifies the example outputted in the python file.


### Requirements

All pip packages needed can be found in requirements.txt

### How to Run

1. Download the dataset: http://cs231n.stanford.edu/tiny-imagenet-200.zip
2. Extract the dataset and place in the current working directory
3. Run the python file: python3 main.py
4. Convert the tflite model to a C header:
    * apt-get install xxd
    * xxd -i etinynet_int8.tflite > model.h 
    * sed -i 's/unsigned char/const unsigned char/g' model.h
    * sed -i 's/const/alignas(8) const/g' model.h
5. Run the arduino C file
