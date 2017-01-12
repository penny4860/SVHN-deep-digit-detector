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
    img_files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=1000, random_order=False)
    annotator = ann.SvhnAnnotation(ANNOTATION_FILE)
    
    # 2. create detector
    det = detector.Detector(model_filename, model_input_shape, rp.MserRegionProposer(), preproc.GrayImgPreprocessor(mean_value))
 
    # 3. Evaluate average precision     
    evaluator = eval.Evaluator(det, annotator, rp.OverlapCalculator())
    recall, precision, f1_score = evaluator.run(img_files)
    recall value : 0.487344684768, precision value : 0.656133828996, f1_score : 0.559281753367
    

    
#     det = detector.Detector(None, mean_value, model_input_shape, rp.MserRegionProposer(), preproc.GrayImgPreprocessor())
#     evaluator = eval.Evaluator(det, annotator, rp.OverlapCalculator())
#     recall, precision, f1_score = evaluator.run(img_files)








