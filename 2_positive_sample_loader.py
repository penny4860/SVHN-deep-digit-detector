#-*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
import progressbar

import digit_detector.region_proposal as rp
import digit_detector.file_io as file_io
import digit_detector.annotation as ann
import digit_detector.show as show
import digit_detector.eval as eval
import digit_detector.utils as utils


N_IMAGES = None
DIR = '../datasets/svhn/train'

# 1. file 을 load
files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
annotation_file = "../datasets/svhn/train/digitStruct.json"
detector = rp.MserDetector()

positive_samples = []
positive_labels = []

bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',], maxval=len(files)).start()

for i, image_file in enumerate(files):
    print image_file,
    image = cv2.imread(image_file)

    gt_bbs, labels = ann.get_annotation(image_file, annotation_file)
    
    for bb, label in zip(gt_bbs, labels):
        patches = utils.crop_bb(image, bb, pad_size=(0, 0), dst_size=(32, 32))
        positive_samples.append(patches)
        positive_labels.append(label)
    
    bar.update(i)

bar.finish()


positive_samples = np.array(positive_samples)
positive_labels = np.array(positive_labels).reshape(-1,1)
print positive_samples.shape

file_io.FileHDF5().write(positive_samples, "positive_images.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(positive_labels, "positive_images.hdf5", "labels", "a", dtype="int")

show.plot_images(positive_samples)
    
    
    





