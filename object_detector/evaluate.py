#-*- coding: utf-8 -*-
import object_detector.file_io as file_io
import cv2
import numpy as np
import object_detector.utils as utils
import matplotlib.pyplot as plt
import progressbar

class Evaluator(object):
    
    def __init__(self):
        self._recall_precision = None
        self._dataset = None
    
    def eval_average_precision(self, test_image_files, 
                               annotation_path, 
                               detector, 
                               window_dim, window_step, pyramid_scale):
        
        """Public function to calculate average precision of the detector.

        Parameters
        ----------
        test_image_files : list of str
            list of test image filenames to evaluate detector's performance
    
        annotation_path : str
            annotation directory path for test_image_files
        
        detector : Detector
            instance of Detector class
        
        window_dim : list
            (height, width) order of sliding window size
            
        window_step : list
            (height_step, width_step) order of sliding window step
            
        pyramid_scale : float
            scaling ratio of building image pyramid
            
        Returns
        ----------
        average_precision : float
            evaluated score for the detector and test images on average precision. 
    
        Examples
        --------
        """
        
        patches = []
        probs = []
        gts = []
        
        # setup the progress bar
        widgets = ["Running for each Test image as gathering patches and its probabilities: ", 
                   progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(test_image_files), widgets=widgets).start()
        
        for i, image_file in enumerate(test_image_files):
            test_image = cv2.imread(image_file)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            boxes, probs_ = detector.run(test_image, window_dim, window_step, pyramid_scale, threshold_prob=0.0)
              
            truth_bb = self._get_truth_bb(image_file, annotation_path)
            ious = self._calc_iou(boxes, truth_bb)
            is_positive = ious > 0.5
             
            patches += boxes.tolist()
            probs += probs_.tolist()
            gts += is_positive.tolist()
            
            pbar.update(i)
        pbar.finish()
    
        probs = np.array(probs)
        gts = np.array(gts)

        self._calc_precision_recall(probs, gts)
        average_precision = self._calc_average_precision()
        
        return average_precision
    
    def plot_recall_precision(self):
        """Function to plot recall-precision graph.
        
        It should be performed eval_average_precision() before this function is called.
        """
        range_offset = 0.1
        
        if self._recall_precision is None:
            raise ValueError('Property _recall_precision is not calculated. To calculate this, run eval_average_precision() first.')
        
        recall_precision = self._recall_precision
        
        plt.plot(recall_precision[:, 0], recall_precision[:, 1], "r-")
        plt.plot(recall_precision[:, 0], recall_precision[:, 1], "ro")
        plt.axis([0 - range_offset, 1 + range_offset, 0 - range_offset, 1 + range_offset])
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.show()
    
    @property
    def dataset(self):
        if self._dataset is None:
            raise ValueError('Property _dataset is not calculated. To calculate this, run eval_average_precision() first.')
        return self._dataset
    
    def _calc_average_precision(self):
        
        inter_precisions = []
        for i in range(11):
            recall = float(i) / 10
            inter_precisions.append(self._calc_interpolated_precision(recall))
            
        return np.array(inter_precisions).mean()

    
    def _calc_precision_recall(self, probs, ground_truths):
        probs = np.array(probs)
        ground_truths = np.array(ground_truths)
        
        dataset = np.concatenate([probs.reshape(-1,1), ground_truths.reshape(-1,1)], axis=1)
        dataset = dataset[dataset[:, 0].argsort()[::-1]]
        
        n_gts = len(dataset[dataset[:, 1] == 1])
        n_relevant = 0.0
        n_searched = 0.0
        
        recall_precision = []
        
        for data in dataset:
            n_searched += 1
            if data[1] == 1:
                n_relevant += 1
            recall = n_relevant / n_gts
            precision = n_relevant / n_searched
            recall_precision.append((recall, precision))
            
            if recall == 1.0:
                break
        
        self._dataset = dataset
        self._recall_precision = np.array(recall_precision)
    
    def _calc_interpolated_precision(self, desired_recall):
        recall_precision = self._recall_precision
        
        inter_precision = recall_precision[recall_precision[:,0] >= desired_recall]
        inter_precision = inter_precision[:, 1]
        inter_precision = max(inter_precision)
        return inter_precision
    
    def _calc_iou(self, boxes, truth_box):
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]
        
        y1_gt = truth_box[0]
        y2_gt = truth_box[1]
        x1_gt = truth_box[2]
        x2_gt = truth_box[3]
        
        xx1 = np.maximum(x1, x1_gt)
        yy1 = np.maximum(y1, y1_gt)
        xx2 = np.minimum(x2, x2_gt)
        yy2 = np.minimum(y2, y2_gt)
    
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersections = w*h
        As = (x2 - x1 + 1) * (y2 - y1 + 1)
        B = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
        
        ious = intersections.astype(float) / (As + B -intersections)
        return ious


    # Todo : extractor module과 중복되는 내용 제거
    def _get_truth_bb(self, image_file, annotation_path):
        image_id = utils.get_file_id(image_file)
        annotation_file = "{}/annotation_{}.mat".format(annotation_path, image_id)
        bb = file_io.FileMat().read(annotation_file)["box_coord"][0]
        return bb

        
