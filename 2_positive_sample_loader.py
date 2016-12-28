import numpy as np
import scipy.io as io
import cv2
import digit_detector.file_io as file_io

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

file_io.FileHDF5().write(images, "positive_images.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(labels, "positive_images.hdf5", "labels", "a", dtype="int")

features = file_io.FileHDF5().read("positive_images.hdf5", "images")
labels_ = file_io.FileHDF5().read("positive_images.hdf5", "labels")

if np.array_equal(images, features):
    print "True"
else:
    print "False"

if np.array_equal(labels_, labels):
    print "True"
else:
    print "False"

print images.shape, labels.shape


