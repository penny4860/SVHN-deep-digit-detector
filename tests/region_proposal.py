
import numpy as np
import cv2
from matplotlib import pyplot as plt


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
            
        return bbs
    
    def _plot(self, img, regions):
        import show
        show.plot_contours(img, regions)
        




