#-*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import digit_detector.extractor as extractor_
import digit_detector.file_io as file_io
import digit_detector.annotation as ann
import digit_detector.show as show
import digit_detector.region_proposal as rp

N_IMAGES = 4
DIR = '../datasets/svhn/train'
NEG_OVERLAP_THD = 0.05
POS_OVERLAP_THD = 0.6
PATCH_SIZE = (32,32)

if __name__ == "__main__":

    # 1. file 을 load
    files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
    annotation_file = "../datasets/svhn/train/digitStruct.json"

    extractor = extractor_.Extractor(rp.MserRegionProposer(), ann.SvhnAnnotation(annotation_file))
    samples, labels = extractor.extract_patch(files, PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)
    print samples.shape, labels.shape
     
    show.plot_images(samples, labels.reshape(-1,).tolist())
     
      
     




