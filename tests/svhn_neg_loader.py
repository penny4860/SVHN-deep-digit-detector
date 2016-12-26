#-*- coding: utf-8 -*-

import region_proposal as rp
import file_io
import cv2
import numpy as np
import os
import annotation as ann
import show
import eval

import matplotlib.pyplot as plt

N_IMAGES = 2
DIR = '../../datasets/svhn/train'

# 1. file 을 load
files = file_io.list_files(directory='../../datasets/svhn/train', n_files_to_sample=N_IMAGES, random_order=False)
annotation_file = "../../datasets/svhn/train/digitStruct.json"
detector = rp.MserDetector()

negative_samples = []
for image_file in files:
    image = cv2.imread(image_file)
    candidates = detector.detect(image, False)
    
    gts, _ = ann.get_annotation(image_file, annotation_file)
    #show.plot_bounding_boxes(image, gts)

    # gts, candidates 의 overlap 을 구한다.
    overlaps = eval.calc_iou(candidates, gts)
    #show.plot_bounding_boxes(image, candidates[overlaps>0.05])

    # Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장
    negative_boxes = candidates[overlaps<0.05]
    
    # negative box 를 crop, resize to 32x32 해서 sample 에 추가
    import utils
    for bb in negative_boxes:
        sample = utils.crop_bb(image, bb, padding=0, dst_size=(32,32))
        negative_samples.append(sample)

negative_samples = np.array(negative_samples)    
print negative_samples.shape

show.plot_images(negative_samples)
    
    
    





