import digit_detector.file_io as file_io
import numpy as np
import os
import cv2

import digit_detector.preprocess as preproc
import digit_detector.train as train_

# DIR = '/datasets/svhn'
NB_FILTERS = 32
NB_EPOCH = 5
BATCHSIZE = 64 

DETECTOR_FILE = 'detector_model.hdf5'
RECOGNIZER_FILE = 'recognize_model.hdf5'

if __name__ == "__main__":

    images_train = file_io.FileHDF5().read("train.hdf5", "images")
    labels_train = file_io.FileHDF5().read("train.hdf5", "labels")

    images_val = file_io.FileHDF5().read("val.hdf5", "images")
    labels_val = file_io.FileHDF5().read("val.hdf5", "labels")

    # Train detector
    X_train, X_val, Y_train, Y_val, mean_value = preproc.GrayImgTrainPreprocessor().run(images_train, labels_train, images_val, labels_val, 2)
    print("mean value of the train images : {}".format(mean_value))    # 108.784
    print("Train image shape is {}, and Validation image shape is {}".format(X_train.shape, X_val.shape))    # (1279733, 32, 32, 1), (317081, 32, 32, 1)
    train_.train_detector(X_train, X_val, Y_train, Y_val, nb_filters = NB_FILTERS, nb_epoch=NB_EPOCH, batch_size=BATCHSIZE, nb_classes=2, save_file=DETECTOR_FILE)
    # loss: 0.0784 - accuracy: 0.9744 - val_loss: 0.0997 - val_accuracy: 0.9724
    # Test score: 0.09970200061798096
    # Test accuracy: 0.9724171161651611
    
    # Train recognizer
    X_train, X_val, Y_train, Y_val, mean_value = preproc.GrayImgTrainPreprocessor().run(images_train, labels_train, images_val, labels_val, 10)
    print("mean value of the train images : {}".format(mean_value))    # 115.503
    print("Train image shape is {}, and Validation image shape is {}".format(X_train.shape, X_val.shape))    # (267234, 32, 32, 1), (67359, 32, 32, 1)
    train_.train_detector(X_train, X_val, Y_train, Y_val, nb_filters = NB_FILTERS, nb_epoch=NB_EPOCH, batch_size=BATCHSIZE, nb_classes=10, save_file=RECOGNIZER_FILE)
    # loss: loss: 0.1070 - accuracy: 0.9685 - val_loss: 0.2196 - val_accuracy: 0.9532
    # Test score: 0.21958307921886444
    # Test accuracy: 0.9531614184379578