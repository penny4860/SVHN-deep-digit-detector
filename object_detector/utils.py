#-*- coding: utf-8 -*-
import cv2
import sklearn.feature_extraction.image as skimg

def crop_bb(image, bb, padding=10, dst_size=(32, 32)):
    
    h = image.shape[0]
    w = image.shape[0]

    (y1, y2, x1, x2) = bb
    
    (x1, y1) = (max(x1 - padding, 0), max(y1 - padding, 0))
    (x2, y2) = (min(x2 + padding, w), min(y2 + padding, h))
    
    roi = image[y1:y2, x1:x2]
    roi = cv2.resize(roi, dst_size, interpolation=cv2.INTER_AREA)
 
    return roi

def crop_random(image, dst_size=(32, 32), max_patches=5):
    
    patches = skimg.extract_patches_2d(image, 
                                       dst_size,
                                       max_patches=max_patches)
    return patches


if __name__ == "__main__":

    import numpy as np
    one_image = np.arange(100).reshape((10, 10))
    
    print one_image
    patches = crop_random(one_image, (4,4), 4)
    
    print patches[0]
    print patches[1]
    print patches[2]
    print patches[3]
    
    



    

