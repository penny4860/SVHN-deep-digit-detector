
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
        
        n_regions = len(regions)
        n_rows = int(np.sqrt(n_regions)) + 1
        n_cols = int(np.sqrt(n_regions)) + 2
        
        # plot original image 
        plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
        plt.imshow(img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
          
        for i, region in enumerate(regions):
            clone = img.copy()
            cv2.drawContours(clone, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
            cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 1)
            plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
            plt.title('Contours'), plt.xticks([]), plt.yticks([])
         
        plt.show()




