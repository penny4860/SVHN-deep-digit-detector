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


N_IMAGES = 2
DIR = '../datasets/svhn/train'
OVERLAP_THD = 0.05

# 1. file 을 load
files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
annotation_file = "../datasets/svhn/train/digitStruct.json"
detector = rp.MserRegionProposer()                      # todo : interface 에 의존하도록 수정하자.
annotator = ann.SvhnAnnotation(annotation_file)  # todo : interface 에 의존하도록 수정하자.
iou_calculator = rp.IouCalculator()

negative_samples = []
positive_samples = []
positive_labels = []

bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',], maxval=len(files)).start()

for i, image_file in enumerate(files):
    print image_file,
    image = cv2.imread(image_file)

    candidate_regions = detector.detect(image)
    patches = candidate_regions.get_patches(0, 0, dst_size=(32,32))
    
    true_boxes, labels = annotator.get_boxes_and_labels(image_file)
    truth_regions = rp.Regions(image, true_boxes)
    truth_patches = truth_regions.get_patches(0, 0, dst_size=(32,32))

    ious, ious_max = iou_calculator.calc(candidate_regions.get_boxes(), truth_regions.get_boxes())
#     show.plot_bounding_boxes(image, candidate_regions.get_boxes())
#     show.plot_bounding_boxes(image, candidate_regions.get_boxes()[overlaps<OVERLAP_THD]) #negative sample plot
  
    # Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장
    negative_samples.append(candidate_regions.get_patches(0, 0, (32,32))[ious_max<OVERLAP_THD])
    
    for i, label in enumerate(labels):
        samples = candidate_regions.get_patches(0, 0, (32,32))[ious[:,i]>0.5]
        labels = np.zeros((len(samples), 1)) + label
        positive_samples.append(samples)
        positive_labels.append(labels)
     
    positive_samples.append(truth_regions.get_patches(0, 0, (32,32)))
    positive_labels.append(labels)
    
    bar.update(i)

bar.finish()


negative_samples = np.concatenate(negative_samples, axis=0)    
negative_labels = np.zeros((len(negative_samples), 1))
positive_samples = np.concatenate(positive_samples, axis=0)    
positive_labels = np.concatenate(positive_labels, axis=0)


print negative_samples.shape, positive_samples.shape
print negative_labels.shape, positive_labels.shape
 
# file_io.FileHDF5().write(negative_samples, "negative_images.hdf5", "images", "w", dtype="uint8")
# file_io.FileHDF5().write(labels, "negative_images.hdf5", "labels", "a", dtype="int")

# show.plot_images(negative_samples)
    
    
    





