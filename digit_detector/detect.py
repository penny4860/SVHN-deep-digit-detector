#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.show as show

class NonMaxSuppressor:
    def __init__(self):
        pass
    
    def run(self, boxes, patches, probs, overlap_threshold=0.3):
        """
        Parameters:
            boxes (ndarray of shape (N, 4))
            patches (ndarray of shape (N, 32, 32, 1))
            probs (ndarray of shape (N,))
            overlap_threshold (float)
        
        Reference: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        """
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes, dtype="float")
        probs = np.array(probs)
     
        pick = []
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]
     
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(probs)
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value to the list of
            # picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            # find the largest (x, y) coordinates for the start of the bounding box and the
            # smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
    
            # delete all indexes from the index list that have overlap greater than the
            # provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
            
        # return only the bounding boxes that were picked
        return boxes[pick].astype("int"), patches[pick], probs[pick]
    

class DigitSpotter:
    
    # Todo classifier, recognizer 정리
    def __init__(self, classifier, recognizer, region_proposer):
        """
        Parameters:
            model_file (str)
            region_proposer (MserRegionProposer)
        """
        self._cls = classifier
        self._recognizer = recognizer
        self._region_proposer = region_proposer
        
    
    def run(self, image, threshold=0.7, do_nms=True, show_result=True, nms_threshold=0.3):
        """Public function to run the DigitSpotter.

        Parameters
        ----------
        image : str
            filename of the test image
            
        Returns
        ----------
        bbs : ndarray, shape of (N, 4)
            detected bounding box. (y1, y2, x1, x2) ordered.
        
        probs : ndarray, shape of (N,)
            evaluated score for the DigitSpotter and test images on average precision. 
    
        Examples
        --------
        """
        
        # 1. Get candidate patches
        candidate_regions = self._region_proposer.detect(image)
        patches = candidate_regions.get_patches(dst_size=self._cls.input_shape)
        
        # 3. Run pre-trained classifier
        probs = self._cls.predict_proba(patches)[:, 1]
    
        # 4. Thresholding
        bbs, patches, probs = self._get_thresholded_boxes(candidate_regions.get_boxes(), patches, probs, threshold)
    
        # 5. non-maxima-suppression
        if do_nms and len(bbs) != 0:
            bbs, patches, probs = NonMaxSuppressor().run(bbs, patches, probs, nms_threshold)
        
        if len(patches) > 0:
            probs_ = self._recognizer.predict_proba(patches)
            y_pred = probs_.argmax(axis=1)
        
        if show_result:
            for i, bb in enumerate(bbs):
                
                # todo : show module 정리
                image = show.draw_box(image, bb, 2)
                
                y1, y2, x1, x2 = bb
                msg = "{0}".format(y_pred[i])
                cv2.putText(image, msg, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
                
            cv2.imshow("MSER + CNN", image)
            cv2.waitKey(0)
        
        return bbs, probs


    def _get_thresholded_boxes(self, bbs, patches, probs, threshold):
        """
        Parameters:
            regions (Regions)
        """
        bbs = bbs[probs > threshold]
        patches = patches[probs > threshold]
        probs = probs[probs > threshold]
        return bbs, patches, probs


if __name__ == "__main__":
    pass






