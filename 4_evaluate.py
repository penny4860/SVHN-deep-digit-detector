#-*- coding: utf-8 -*-

# Todo : keras model 에서 predict_probs() 할 때 message off 하는 방법
# evaluator.run(img_files, do_nms=False) 에서 do_nms option 을 사용하지 않도록 detector 자체의 class 에서 nms 객체를 갖고 있도록 하자.

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
    classifier = detector.CnnClassifier(model_filename, model_input_shape)
    preprocessor = preproc.GrayImgPreprocessor(mean_value)
    proposer = rp.MserRegionProposer()
    
    # 2. create detector
    det = detector.Detector(classifier, proposer, preprocessor)
  
    # 3. Evaluate average precision     
    evaluator = eval.Evaluator(det, annotator, rp.OverlapCalculator())
    recall, precision, f1_score = evaluator.run(img_files)
    # recall value : 0.513115508514, precision value : 0.714285714286, f1_score : 0.597214783074
    
    # 4. Evaluate MSER
    classifier = detector.TrueBinaryClassifier(input_shape=(32,32))
    preprocessor = preproc.NonePreprocessor()
    det = detector.Detector(classifier, proposer, preprocessor)
    evaluator = eval.Evaluator(det, annotator, rp.OverlapCalculator())
    recall, precision, f1_score = evaluator.run(img_files, do_nms=False)
    #recall value : 0.630004601933, precision value : 0.0452547023239, f1_score : 0.0844436220084






