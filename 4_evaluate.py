#-*- coding: utf-8 -*-

# Todo : keras model 에서 predict_probs() 할 때 message off 하는 방법
# evaluator.run(img_files, do_nms=False) 에서 do_nms option 을 사용하지 않도록 detector 자체의 class 에서 nms 객체를 갖고 있도록 하자.

import cv2
import numpy as np

import digit_detector.region_proposal as rp
import digit_detector.detect as detect
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.annotation as ann
import digit_detector.evaluate as eva
import digit_detector.classify as cls

model_filename = "detector_model.hdf5"
model_input_shape = (32,32,1)
DIR = 'C:/Users/zhan14417/OneDrive/GitHub/SVHN-deep-digit-detector/datasets/svhn/train'
ANNOTATION_FILE = "C:/Users/zhan14417/OneDrive/GitHub/SVHN-deep-digit-detector/annotation/test/digitStruct.json"

detect_model = "detector_model.hdf5"
recognize_model = "recognize_model.hdf5"
mean_value_for_detector = 108.784
mean_value_for_recognizer = 115.503


if __name__ == "__main__":
    # 1. load test image files, annotation file
    img_files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=1000, random_order=False)
    annotator = ann.SvhnAnnotation(ANNOTATION_FILE)
    
    preprocessor_for_detector = preproc.GrayImgPreprocessor(mean_value_for_detector)
    preprocessor_for_recognizer = preproc.GrayImgPreprocessor(mean_value_for_recognizer)

    detector = cls.CnnClassifier(detect_model, preprocessor_for_detector, model_input_shape)
    recognizer = cls.CnnClassifier(recognize_model, preprocessor_for_recognizer, model_input_shape)

    proposer = rp.MserRegionProposer()
    
    # 2. create detector
    det = detect.DigitSpotter(detector, recognizer, proposer)
     
    # 3. Evaluate average precision     
    evaluator = eva.Evaluator(det, annotator, rp.OverlapCalculator())
    recall, precision, f1_score = evaluator.run(img_files)
    # recall value : 0.025614754098360656, precision value : 0.0328515111695138, f1_score : 0.02878526194588371
    
    # 4. Evaluate MSER
    detector = cls.TrueBinaryClassifier(input_shape=model_input_shape)
    preprocessor = preproc.NonePreprocessor()
     
    # Todo : detector, recognizer 를 none type 으로
    det = detect.DigitSpotter(detector, recognizer, proposer)
    evaluator = eva.Evaluator(det, annotator, rp.OverlapCalculator())
    recall, precision, f1_score = evaluator.run(img_files, do_nms=False)
    # recall value : 0.08504098360655737, precision value : 0.001661811374398094, f1_score : 0.0032599196802922145