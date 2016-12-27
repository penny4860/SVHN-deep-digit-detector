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
print images.shape, labels.shape
# (73257L, 32L, 32L, 3L) (73257L, 1L)

import digit_detector.file_io as file_io

#     def write(self, data, filename, db_name, write_mode="a"):

file_io.FileHDF5().write(images, "dataset.hdf5", "positive")








