#-*- coding: utf-8 -*-

import cv2
import numpy as np

import digit_detector.region_proposal as rp
import digit_detector.detect as detector
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.annotation as ann

import digit_detector.evaluate as eval
        
    
model_filename = "detector_model.hdf5"
mean_value = 107.524
model_input_shape = (32,32,1)
DIR = '../datasets/svhn/train'
ANNOTATION_FILE = "../datasets/svhn/train/digitStruct.json"

if __name__ == "__main__":
    # 1. load test image files, annotation file
    img_files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=2, random_order=False)
    annotator = ann.SvhnAnnotation(ANNOTATION_FILE)
    
#     # 2. create detector
#     det = detector.Detector(model_filename, mean_value, model_input_shape, rp.MserRegionProposer(), preproc.GrayImgPreprocessor())
# 
#     # 3. Evaluate average precision     
#     evaluator = eval.Evaluator(det, annotator)
#     recall, precision, f1_score = evaluator.run(img_files)
#     # ecall value : 0.45329038196, precision value : 0.63141025641, f1_score : 0.527725689794
    
    det = detector.Detector(None, mean_value, model_input_shape, rp.MserRegionProposer(), preproc.GrayImgPreprocessor())
    evaluator = eval.Evaluator(det, annotator)
    recall, precision, f1_score = evaluator.run(img_files)








