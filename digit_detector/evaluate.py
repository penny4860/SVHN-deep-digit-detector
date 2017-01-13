#-*- coding: utf-8 -*-

import progressbar
import cv2
import numpy as np
import region_proposal as rp


class Evaluator(object):
    
    def __init__(self, detector, annotator, overlap_calculator):
        """
        detector : Detector
            instance of Detector class
        
        overlap_calculator : OverlapCalculator
            instance of OverlapCalculator class
        
        """
        self._detector = detector
        self._annotator = annotator
        self._overlap_calculator = overlap_calculator


    def run(self, test_image_files, do_nms=True):
        # Todo : nms 를 삭제
        
        # setup the progress bar
        widgets = ["Running for each Test image as gathering patches and its probabilities: ", 
                   progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(test_image_files), widgets=widgets).start()
        
        n_detected = 0
        n_truth = 0
        n_true_positive = 0
        
        for i, image_file in enumerate(test_image_files):
            test_image = cv2.imread(image_file)
            
            # 1. Get the detected boxes
            detected_bbs, detected_probs_ = self._detector.run(test_image, threshold=0.5, do_nms=do_nms, nms_threshold=0.1, show_result=False)

            # 2. Get the true boxes
            true_bbs, true_labels = self._annotator.get_boxes_and_labels(image_file)

            # 3. Calc IOU between detected and true boxes
            overlaps_per_truth = self._overlap_calculator.calc_ious_per_truth(detected_bbs, true_bbs)

            n_true_positive += self._calc_true_positive(overlaps_per_truth)
            n_detected += len(detected_bbs)
            n_truth += len(true_bbs)
             
            pbar.update(i)
        pbar.finish()

        recall = float(n_true_positive) / n_truth
        precision = float(n_true_positive) / n_detected
        f1_score = 2* (precision*recall) / (precision + recall)
        self._print_msg(recall, precision, f1_score)
        
        return recall, precision, f1_score
    

    def _calc_true_positive(self, overlaps_per_truth):
        """
        Parameters:
            overlaps_per_truth (N, M)
                N : number of Truth
                M : number of Detected
        """
        n_true_positive = 0
        for overlaps in overlaps_per_truth:
            if len(overlaps) > 0 and overlaps.max() > 0.5:
                n_true_positive += 1
        return n_true_positive

    
    def _print_msg(self, recall, precision, f1_score):
        print "recall value : {}, precision value : {}, f1_score : {}".format(recall, precision, f1_score)
