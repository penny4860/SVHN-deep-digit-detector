#-*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

import crop
import show


class Regions:
    
    """ Image 에서의 bounding box 를 관리하는 class """
    
    def __init__(self, image, boxes, cropper=crop.CropperWithoutPad()):
        self._image = image
        self._boxes = boxes
        self._cropper = cropper
    
    def get_boxes(self):
        return self._boxes
    
    def get_patches(self, dst_size=None):
        patches = []
        for bb in self._boxes:
            patch = self._crop(bb)
            
            if dst_size:
                desired_ysize = dst_size[0]
                desired_xsize = dst_size[1]
                patch = cv2.resize(patch, (desired_xsize, desired_ysize), interpolation=cv2.INTER_AREA)
                
            patches.append(patch)
            
        if dst_size:
            return np.array(patches)
        else:
            return patches
    
    def _crop(self, box):
        patch = self._cropper.crop(self._image, box)
        return patch

class _RegionProposer:
    
    def __init__(self):
        pass
    
    def detect(self, img):
        pass

    def _to_gray(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray

    
class MserRegionProposer(_RegionProposer):
    
    def detect(self, img):
        gray = self._to_gray(img)
        mser = cv2.MSER(_delta = 1)
        regions = mser.detect(gray, None)
        bounding_boxes = self._get_boxes(regions)
        regions = Regions(img, bounding_boxes)
        return regions
    
    def _get_boxes(self, regions):
        bbs = []
        for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
            bbs.append((y, y+h, x, x+w))
            
        return np.array(bbs)


class OverlapCalculator:
    
    def __init__(self):
        pass
    
    def calc_ious_per_truth(self, boxes, true_boxes):
        return self._calc(boxes, true_boxes)
    
    def calc_maximun_ious(self, boxes, true_boxes):
        ious_for_each_gt = self._calc(boxes, true_boxes)
        ious = np.max(ious_for_each_gt, axis=0)
        return ious
    
    def _calc(self, boxes, true_boxes):
        ious_for_each_gt = []
        
        for truth_box in true_boxes:
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
            ious_for_each_gt.append(ious)
        
        # (n_truth, n_boxes)
        ious_for_each_gt = np.array(ious_for_each_gt)
        return ious_for_each_gt
        



