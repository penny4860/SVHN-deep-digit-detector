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


N_IMAGES = 10
DIR = '../datasets/svhn/train'
NEG_OVERLAP_THD = 0.05
POS_OVERLAP_THD = 0.6
PATCH_SIZE = (32,32)


class Extractor:
    
    def __init__(self):
        pass
    
    def extract_patch(self, image_files, patch_size, positive_overlap_thd, negative_overlap_thd):
        detector = rp.MserRegionProposer()                      # todo : interface 에 의존하도록 수정하자.
        annotator = ann.SvhnAnnotation(annotation_file)  # todo : interface 에 의존하도록 수정하자.
        
        negative_samples = []
        positive_samples = []
        positive_labels = []
        
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',], maxval=len(image_files)).start()
    
        for i, image_file in enumerate(image_files):
            print image_file,
            image = cv2.imread(image_file)
         
            candidate_regions = detector.detect(image)
            candidate_patches = candidate_regions.get_patches(dst_size=PATCH_SIZE)
             
            true_boxes, labels = annotator.get_boxes_and_labels(image_file)
            truth_regions = rp.Regions(image, true_boxes)
            truth_patches = truth_regions.get_patches(dst_size=PATCH_SIZE)
         
            ious, ious_max = rp.calc_overlap(candidate_regions.get_boxes(), truth_regions.get_boxes())
           
            # Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장
            negative_samples.append(candidate_patches[ious_max<negative_overlap_thd])
             
            for i, label in enumerate(labels):
                samples = candidate_patches[ious[i,:]>positive_overlap_thd]
                labels_ = np.zeros((len(samples), )) + label
                positive_samples.append(samples)
                positive_labels.append(labels_)
                 
            positive_samples.append(truth_regions.get_patches(PATCH_SIZE))
            positive_labels.append(labels)
            bar.update(i)
        bar.finish()
         
        negative_samples = np.concatenate(negative_samples, axis=0)    
        negative_labels = np.zeros((len(negative_samples), 1))
        positive_samples = np.concatenate(positive_samples, axis=0)    
        positive_labels = np.concatenate(positive_labels, axis=0)
        
        return positive_samples, positive_labels, negative_samples, negative_labels
    
    def _select_positive_patch(self):
        pass
    
    def _select_negative_patch(self):
        pass



if __name__ == "__main__":

    # 1. file 을 load
    files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
    annotation_file = "../datasets/svhn/train/digitStruct.json"
    
    positive_samples, positive_labels, negative_samples, negative_labels = Extractor().extract_patch(files, PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)
         
    print negative_samples.shape, positive_samples.shape
    print negative_labels.shape, positive_labels.shape
     
#     show.plot_images(positive_samples, positive_labels.tolist())
#     show.plot_images(negative_samples)
     
      
    # file_io.FileHDF5().write(negative_samples, "negative_images.hdf5", "images", "w", dtype="uint8")
    # file_io.FileHDF5().write(labels, "negative_images.hdf5", "labels", "a", dtype="int")
     




