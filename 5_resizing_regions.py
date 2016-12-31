#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.detect as detector
import digit_detector.file_io as file_io


model_filename = "detector_model.hdf5"
mean_value = 108.546
DIR = '../datasets/svhn/train'

if __name__ == "__main__":
    # 1. image files
    img_files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=None, random_order=False)
    
    for img_file in img_files[30:40]:
        # 2. image
        img = cv2.imread(img_file)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detector.detect(img, model_filename, mean_value, threshold=0.5, do_nms=False)







