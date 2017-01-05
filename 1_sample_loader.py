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


N_IMAGES = 4
DIR = '../datasets/svhn/train'
NEG_OVERLAP_THD = 0.05
POS_OVERLAP_THD = 0.6
PATCH_SIZE = (32,32)


class Extractor:
    
    def __init__(self):
        self._positive_samples = []
        self._negative_samples = []
        self._positive_labels = []
        self._negative_labels = []
    
    def extract_patch(self, image_files, patch_size, positive_overlap_thd, negative_overlap_thd):
        detector = rp.MserRegionProposer()                      # todo : interface 에 의존하도록 수정하자.
        annotator = ann.SvhnAnnotation(annotation_file)  # todo : interface 에 의존하도록 수정하자.
        
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',], maxval=len(image_files)).start()
    
        for i, image_file in enumerate(image_files):
            print image_file,
            image = cv2.imread(image_file)
         
            # 1. detect regions
            candidate_regions = detector.detect(image)
            candidate_patches = candidate_regions.get_patches(dst_size=patch_size)
            candidate_boxes = candidate_regions.get_boxes()
             
            # 2. load ground truth
            true_boxes, true_labels = annotator.get_boxes_and_labels(image_file)
            true_patches = rp.Regions(image, true_boxes).get_patches(dst_size=patch_size)
            
            # 3. calc overlap
            overlaps = rp.calc_overlap(candidate_boxes, true_boxes)

            # 4. add patch to the samples
            self._select_positive_patch(candidate_patches, true_labels, overlaps, positive_overlap_thd)
            self._append_positive_patch(true_patches, true_labels)
            self._select_negative_patch(candidate_patches, overlaps, negative_overlap_thd)
           
            bar.update(i)
        bar.finish()
         
        return self._merge_sample()
    
    # Todo : positive patch, negative patch 를 선택하는 부분을 따로 모듈화할 수 있을지 생각해보자. 
    def _append_positive_patch(self, true_patches, true_labels):
        self._positive_samples.append(true_patches)
        self._positive_labels.append(true_labels)
        
    def _select_positive_patch(self, candidate_patches, true_labels, overlaps, overlap_thd):
        for i, label in enumerate(true_labels):
            samples = candidate_patches[overlaps[i,:]>overlap_thd]
            labels_ = np.zeros((len(samples), )) + label
            self._positive_samples.append(samples)
            self._positive_labels.append(labels_)
    
    def _select_negative_patch(self, candidate_patches, overlaps, overlap_thd):
        overlaps_max = np.max(overlaps, axis=0)
        self._negative_samples.append(candidate_patches[overlaps_max<overlap_thd])

    def _merge_sample(self):
        negative_samples = np.concatenate(self._negative_samples, axis=0)    
        negative_labels = np.zeros((len(negative_samples), 1))
        positive_samples = np.concatenate(self._positive_samples, axis=0)    
        positive_labels = np.concatenate(self._positive_labels, axis=0).reshape(-1,1)

        samples = np.concatenate([negative_samples, positive_samples], axis=0)
        labels = np.concatenate([negative_labels, positive_labels], axis=0)
        return samples, labels

if __name__ == "__main__":

    # 1. file 을 load
    files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
    annotation_file = "../datasets/svhn/train/digitStruct.json"
    
    samples, labels = Extractor().extract_patch(files, PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)
    print samples.shape, labels.shape
     
    show.plot_images(samples, labels.reshape(-1,).tolist())
     
      
     




