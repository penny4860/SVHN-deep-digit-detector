import numpy as np
import scipy.io as io
import cv2

N_IMAGES = 2
MAT_FILE = '../datasets/svhn/train_32x32.mat'

def load_svhn_images(file_name):
    dataset = io.loadmat(file_name)
    images = dataset['X']
    labels = dataset['y']
    images = np.transpose(images, (3, 0, 1, 2))
    
    # (N, rows, cols, channels)
    # (N, 1)
    return images, labels


images, labels = load_svhn_images(MAT_FILE)    

import digit_detector.file_io as file_io

file_io.FileHDF5().write(images, "svhn_dataset.hdf5", "features", "w")
file_io.FileHDF5().write(labels, "svhn_dataset.hdf5", "labels", "a")

features = file_io.FileHDF5().read("svhn_dataset.hdf5", "features")
labels_ = file_io.FileHDF5().read("svhn_dataset.hdf5", "labels")

print images.shape, labels.shape
print features.shape

if np.array_equal(images, features):
    print "True"
else:
    print "False"

if np.array_equal(labels_, labels):
    print "True"
else:
    print "False"



