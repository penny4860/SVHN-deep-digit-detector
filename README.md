# SVHN-deep-cnn-digit-detector

This project implements deep-cnn-detector (and recognizer) in natural scene. I used keras framework and opencv library to build the detector.
This detector determine digit or not with CNN classifier for the region proposed by the MSER algorithm.
The procedure to build digit detector is as follows:

### 1. load training samples (1_sample_loader.py)

Svhn provides cropped training samples in matlab format. 
However, it is not suitable for detecting bounding box because it introduces some distracting digits to the sides of the digit of interest. So I collected the training samples directly using full numbers images and its annotation file.

* Train samples : (457723, 32, 32, 3)
* Validation samples : (113430, 32, 32, 3)


### 2. train detector (2_train.py)

I designed a classifier for architecture similar to VGG-Net. 

The architecture is as follows:

* INPUT: [32x32x1]
* CONV3-32: [32x32x32]
* CONV3-32: [32x32x32]
* POOL2: [16x16x32]
* CONV3-64: [16x16x64]
* CONV3-64: [16x16x64]
* POOL2: [8x8x64]
* FC: [1x1x1024] I used drop out in this layer.
* FC: [1x1x2]

The accuracy of the classifier is as follows

* Training Accuracy : 97.91%
* Test Accuracy : 96.98%

### 3. Run the detector (3_detect.py)

In the running time, the detector operates in the 2-steps.

1) The detector finds candidate region proposed by the MSER algorithm.

<img src="examples/mser.png">

2) The classifier determines whether or not it is a number in the proposed region.

<img src="examples/classifier.png">

