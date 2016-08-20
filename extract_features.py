#-*- coding: utf-8 -*-

import numpy as np
import random
import cv2
import object_detector.file_io as file_io
import object_detector.descriptor as descriptor
import object_detector.utils as utils

CONFIGURATION_FILE = "conf/cars.json"
POSITIVE_LABEL_NUMBER = 1
NEGATIVE_LABEL_NUMBER = 0

conf = file_io.FileJson().read(CONFIGURATION_FILE)

# initialize the HOG descriptor along with the list of data and labels
hog = descriptor.HOG(conf['orientations'],
                     conf['pixels_per_cell'],
                     conf['cells_per_block'])

train_image_files = file_io.list_files(conf["image_dataset"], "*.jpg")
train_image_files = random.sample(train_image_files, int(len(train_image_files) * conf["percent_gt_images"]))

negative_image_files = file_io.list_files(conf["image_distractions"], "*.jpg")
negative_image_files = random.sample(negative_image_files, conf["num_distraction_images"])

positive_data = None
negative_data = None

for (i, file_) in enumerate(train_image_files):
    # load the image, convert it to grayscale, and extract the image ID from the path
    image = cv2.imread(file_)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageID = utils.get_file_id(file_)
    
    # load the annotation file associated with the image and extract the bounding box
    annotation_file = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
    bb = file_io.FileMat().read(annotation_file)["box_coord"][0]
    roi = utils.crop_bb(image, bb, padding=conf["offset"], dst_size=tuple(conf["window_dim"]))
    
    # Todo : augment modulization
    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)
    
    features = hog.describe(rois)
    
    if positive_data is None:
        positive_data = features
    else:
        positive_data = np.concatenate([positive_data, features], axis = 0)

# loop over the desired number of distraction images
for (i, file_) in enumerate(negative_image_files):
    image = cv2.imread(file_)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    patches = utils.crop_random(image, tuple(conf["window_dim"]), conf["num_distractions_per_image"])
    features = hog.describe(patches)
    
    if negative_data is None:
        negative_data = features
    else:
        negative_data = np.concatenate([negative_data, features], axis = 0)

positive_labels = np.zeros((len(positive_data), 1)) + POSITIVE_LABEL_NUMBER
negative_labels = np.zeros((len(negative_data), 1)) + NEGATIVE_LABEL_NUMBER
pos_set = np.concatenate([positive_labels, positive_data], axis=1)
neg_set = np.concatenate([negative_labels, negative_data], axis=1)
data = np.concatenate([pos_set, neg_set], axis=0)

print("[INFO] number of positive data : {}, number of negative data : {}".format(positive_data.shape, negative_data.shape))

file_io.FileHDF5().write(data, conf["features_path"], "features")




    
