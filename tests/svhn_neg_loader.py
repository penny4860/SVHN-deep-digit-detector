#-*- coding: utf-8 -*-

import region_proposal as rp
import file_io
import cv2
import os
import annotation as ann

import matplotlib.pyplot as plt

N_IMAGES = 2
DIR = '../../datasets/svhn/train'

# 1. file 을 load
files = file_io.list_files(directory='../../datasets/svhn/train', n_files_to_sample=N_IMAGES, random_order=False)
annotation_file = "../../datasets/svhn/train/digitStruct.json"
detector = rp.MserDetector()


for image_file in files:
    image = cv2.imread(image_file)
    candidates = detector.detect(image, True)
    
    gts, labels = ann.get_annotation(image_file, annotation_file)
    
    for ground_truth in gts:
        y1, y2, x1, x2 = ground_truth
        
        clone = image.copy()
        cv2.rectangle(clone, (x1, y1), (x2, y2), (255, 0, 0), 1)
    plt.imshow(clone)
    plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    

    
    


# 3. digitStruct.json file load



# 3. Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장


