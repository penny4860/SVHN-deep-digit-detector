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
OVERLAP_THD = 0.05

# 1. file 을 load
files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
annotation_file = "../datasets/svhn/train/digitStruct.json"
detector = rp.MserDetector()

negative_samples = []

bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',], maxval=len(files)).start()

for i, image_file in enumerate(files):
    print image_file,
    image = cv2.imread(image_file)
    candidates, bbs = rp.propose_patches(image, dst_size = (32, 32), pad=False)

    gts, _ = ann.get_annotation(image_file, annotation_file)
    #show.plot_bounding_boxes(image, gts)
 
    # gts, candidates 의 overlap 을 구한다.
    overlaps = eval.calc_iou(bbs, gts)
#     show.plot_bounding_boxes(image, bbs[overlaps<0.05]) #negative sample plot
 
    # Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장
    negative_samples.append(candidates[overlaps<OVERLAP_THD, :, :, :])
    bar.update(i)

bar.finish()


negative_samples = np.concatenate(negative_samples, axis=0)    
print negative_samples.shape
labels = np.zeros((len(negative_samples), 1))

file_io.FileHDF5().write(negative_samples, "negative_images.hdf5", "images", "w", dtype="uint8")
file_io.FileHDF5().write(labels, "negative_images.hdf5", "labels", "a", dtype="int")

# show.plot_images(negative_samples)
    
    
    





