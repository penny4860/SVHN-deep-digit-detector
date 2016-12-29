
import numpy as np
import cv2
from matplotlib import pyplot as plt

import show
import utils

class MserDetector:

    def __init__(self):
        pass
    
    def detect(self, img, show_option=False):
        gray = self._to_gray(img)
    
        mser = cv2.MSER(_delta = 1)
        regions = mser.detect(gray, None)
        bounding_boxes = self._get_boxes(regions)
    
        if show_option:
            self._plot(img, regions)

        return bounding_boxes
    
    def _to_gray(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray
    
    def _get_boxes(self, regions):
        bbs = []
        for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
            bbs.append((y, y+h, x, x+w))
            
        return np.array(bbs)
    
    def _plot(self, img, regions):
        show.plot_contours(img, regions)
        
        
def propose_patches(image, dst_size=(32, 32)):
    detector = MserDetector()
    candidates_bbs = detector.detect(image, False)

    patches = []
    for bb in candidates_bbs:
        y1, y2, x1, x2 = bb
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        
        if width >= height:
            pad_y = 0
            pad_x = 0
        else:
            pad_x = int((height-width)/2)
            pad_y = 0
        sample = utils.crop_bb(image, bb, pad_size=(pad_y ,pad_x), dst_size=dst_size)
        patches.append(sample)
    return np.array(patches), candidates_bbs




