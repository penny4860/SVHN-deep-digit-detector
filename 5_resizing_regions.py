#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.detect as detector


img_files = ['imgs/1.png', 'imgs/2.png']
model_filename = "detector_model.hdf5"
mean_value = 108.546


if __name__ == "__main__":
    # 1. image files
    img_file = img_files[1]
    
    # 2. image
    img = cv2.imread(img_file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector.detect(img, model_filename, mean_value)







