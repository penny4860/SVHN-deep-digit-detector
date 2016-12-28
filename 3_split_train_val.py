
import digit_detector.file_io as file_io
import numpy as np
import os

DIR = "../datasets/svhn"

positive_images = file_io.FileHDF5().read(os.path.join(DIR, "positive_images.hdf5"), "images")
positive_labels = file_io.FileHDF5().read(os.path.join(DIR, "positive_images.hdf5"), "labels")

negative_images = file_io.FileHDF5().read(os.path.join(DIR, "negative_images.hdf5"), "images")
negative_labels = file_io.FileHDF5().read(os.path.join(DIR, "negative_images.hdf5"), "labels")

print positive_images.shape
print positive_labels.shape
print negative_images.shape
print negative_labels.shape

# (73257, 32, 32, 3)
# (73257, 1)
# (129298, 32, 32, 3)
# (129298, 1)

N_TRAIN_POS = 60000     #81.9%
N_TRAIN_NEG = 105000    #81.2%

train = {'pos':{'images':None, 'labels':None}, 'neg':{'images':None, 'labels':None}, 'imgs':None, 'labels':None}
val = {'pos':{'images':None, 'labels':None}, 'neg':{'images':None, 'labels':None}, 'imgs':None, 'labels':None}

train['pos']['images'] = positive_images[:N_TRAIN_POS]
train['pos']['labels'] = positive_labels[:N_TRAIN_POS]
val['pos']['images'] = positive_images[N_TRAIN_POS:]
val['pos']['labels'] = positive_labels[N_TRAIN_POS:]

train['neg']['images'] = negative_images[:N_TRAIN_NEG]
train['neg']['labels'] = negative_labels[:N_TRAIN_NEG]
val['neg']['images'] = negative_images[N_TRAIN_NEG:]
val['neg']['labels'] = negative_labels[N_TRAIN_NEG:]


train['imgs'] = np.concatenate([train['pos']['images'], train['neg']['images']], axis=0)
train['labels'] = np.concatenate([train['pos']['labels'], train['neg']['labels']], axis=0)
val['imgs'] = np.concatenate([val['pos']['images'], val['neg']['images']], axis=0)
val['labels'] = np.concatenate([val['pos']['labels'], val['neg']['labels']], axis=0)

print train['imgs'].shape, train['labels'].shape
print val['imgs'].shape, val['labels'].shape


file_io.FileHDF5().write(train['imgs'], "train.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(train['labels'], "train.hdf5", "labels", "a", dtype="int")
file_io.FileHDF5().write(val['imgs'], "val.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(val['labels'], "val.hdf5", "labels", "a", dtype="int")

print "done"
