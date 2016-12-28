
import digit_detector.file_io as file_io
import h5py
import cv2
import numpy as np

images = file_io.FileHDF5().read("positive.hdf5", "features")
labels = file_io.FileHDF5().read("positive.hdf5", "labels")

file_io.FileHDF5().write(images, "positive_images.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(labels, "positive_images.hdf5", "labels", "a", dtype="int")

print images.shape

images = file_io.FileHDF5().read("negative.hdf5", "features")
labels = file_io.FileHDF5().read("negative.hdf5", "labels")

file_io.FileHDF5().write(images, "negative_images.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(labels, "negative_images.hdf5", "labels", "a", dtype="int")

print images.shape


