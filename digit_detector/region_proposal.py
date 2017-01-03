#-*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

import show
import utils


class Regions:
    
    """ Image 에서의 bounding box 를 관리하는 class """
    
    def __init__(self, image, boxes):
        self._image = image
        self._boxes = boxes
    
    def get_boxes(self):
        return self._boxes
    
    def get_patches(self, pad_y, pad_x, dst_size=None):
        patches = []
        for bb in self._boxes:
            patch = self._crop(bb, pad_y ,pad_x)
            
            if dst_size:
                desired_ysize = dst_size[0]
                desired_xsize = dst_size[1]
                patch = cv2.resize(patch, (desired_xsize, desired_ysize), interpolation=cv2.INTER_AREA)
                
            patches.append(patch)
        return np.array(patches)
    
    def _crop(self, box, pad_y, pad_x):
        h = self._image.shape[0]
        w = self._image.shape[1]
        (y1, y2, x1, x2) = box
        
        (x1, y1) = (max(x1 - pad_x, 0), max(y1 - pad_y, 0))
        (x2, y2) = (min(x2 + pad_x, w), min(y2 + pad_y, h))
        patch = self._image[y1:y2, x1:x2]
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



        
        
