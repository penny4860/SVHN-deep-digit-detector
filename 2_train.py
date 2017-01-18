import digit_detector.file_io as file_io
import numpy as np
import os
import cv2

import digit_detector.preprocess as preproc
import digit_detector.train as train_

DIR = ""
NB_FILTERS = 32
NB_EPOCH = 5
SAVE_FILE = 'recognize_model.hdf5'

if __name__ == "__main__":

    images_train = file_io.FileHDF5().read(os.path.join(DIR, "train.hdf5"), "images")
    labels_train = file_io.FileHDF5().read(os.path.join(DIR, "train.hdf5"), "labels")

    images_val = file_io.FileHDF5().read(os.path.join(DIR, "val.hdf5"), "images")
    labels_val = file_io.FileHDF5().read(os.path.join(DIR, "val.hdf5"), "labels")
    
    X_train, X_val, Y_train, Y_val, mean_value = preproc.GrayImgTrainPreprocessor().run(images_train, labels_train, images_val, labels_val, 10)
  
    print "mean value of the train images : {}".format(mean_value)    # 107.524
    print "Train image shape is {}, and Validation image shape is {}".format(X_train.shape, X_val.shape)
    
    train_.train_detector(X_train, X_val, Y_train, Y_val, nb_filters = NB_FILTERS, nb_epoch=NB_EPOCH, nb_classes=10, save_file=SAVE_FILE)
    # acc: 0.9541 - val_loss: 0.2125 - val_acc: 0.9452



