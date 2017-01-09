#-*- coding: utf-8 -*-

import progressbar
import cv2
import numpy as np

import digit_detector.region_proposal as rp
import digit_detector.detect as detector
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.annotation as ann


class Evaluator(object):
    
    def __init__(self, detector, annotator):
        """
        detector : Detector
            instance of Detector class
        """
        self._detector = detector
        self._annotator = annotator

    def calc_precision_and_recall(self, test_image_files):
        
        # setup the progress bar
        widgets = ["Running for each Test image as gathering patches and its probabilities: ", 
                   progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(test_image_files), widgets=widgets).start()
        
        gts = []
        
        n_detected = 0
        n_truth = 0
        n_true_positive = 0
        
        for i, image_file in enumerate(test_image_files):
            test_image = cv2.imread(image_file)
            
            # 1. Get the detected boxes
            detected_bbs, detected_probs_ = self._detector.run(test_image, threshold=0.5, do_nms=True, nms_threshold=0.1, show_result=False)

            # 2. Get the true boxes
            true_bbs, true_labels = annotator.get_boxes_and_labels(image_file)

            # 3. Calc IOU between detected and true boxes
            # Todo : class 로 모듈화하자.
            overlaps = rp.calc_overlap(detected_bbs, true_bbs)
            overlaps = np.max(overlaps, axis=0)
            
            n_detected += len(detected_bbs)
            n_truth += len(true_bbs)
            n_true_positive += len(overlaps[overlaps > 0.5])
             
            pbar.update(i)
        pbar.finish()

        recall = float(n_true_positive) / n_truth
        precision = float(n_true_positive) / n_detected
        
        f1_score = 2* (precision*recall) / (precision + recall)
        
        return recall, precision, f1_score
    
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
    det = detector.Detector(model_filename, mean_value, model_input_shape, rp.MserRegionProposer(), preproc.GrayImgPreprocessor())

    # 3. Evaluate average precision     
    evaluator = Evaluator(det, annotator)
    recall, precision, f1_score = evaluator.calc_precision_and_recall(img_files)
    
    print "recall value : {}, precision value : {}, f1_score : {}".format(recall, precision, f1_score)










