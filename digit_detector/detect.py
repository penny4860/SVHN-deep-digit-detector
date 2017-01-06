#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.region_proposal as rp
import digit_detector.show as show


class Detector:
    
    def __init__(self, model_file, image_mean):
        self._image_mean = image_mean
        self._cls = keras.models.load_model(model_file)
        
        self._region_proposer = rp.MserRegionProposer()
    
    def run(self, image, input_shape = (32,32,1), threshold=0.9, do_nms=True):
        
        candidate_regions = self._region_proposer.detect(image)
        patches = candidate_regions.get_patches(dst_size=(input_shape[0], input_shape[1]))
        
        # 4. Convert to gray
        patches = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches]
        patches = np.array(patches)
        patches = patches.astype('float32')
        patches -= self._image_mean
    
        patches = patches.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
        
        # 5. Run pre-trained classifier
        probs = self._cls.predict_proba(patches)[:, 1]
         
    #     show.plot_images(temp, probs.tolist())
    #     show.plot_bounding_boxes(image, bbs, probs.tolist())
    
        # Thresholding
        bbs = candidate_regions.get_boxes()
        bbs = bbs[probs > threshold]
        probs = probs[probs > threshold]
        
        y1 = bbs[:, 0]
        y2 = bbs[:, 1]
        x1 = bbs[:, 2]
        x2 = bbs[:, 3]
    
        widths = x2-x1+1
        heights = y2-y1+1
        
        probs = probs[heights > widths]
        bbs = bbs[heights > widths]
    
        if do_nms and len(bbs) != 0:
            bbs, probs = self._do_non_max_sup(bbs, probs, 0.1)
    
        for i, bb in enumerate(bbs):
            image = show.draw_box(image, bb, 2)
        cv2.imshow("MSER + CNN", image)
        cv2.waitKey(0)


    def _do_non_max_sup(self, boxes, probs, overlapThresh=0.3):
        """
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
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        # return only the bounding boxes that were picked
        return boxes[pick].astype("int"), probs[pick]


if __name__ == "__main__":
    pass






