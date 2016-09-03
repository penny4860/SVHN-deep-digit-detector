#-*- coding: utf-8 -*-
import os
import numpy as np
import object_detector.descriptor as descriptor
from skimage import data
import cv2


def test_hog_descriptor():

    # Given test image and hog instance
    image = data.camera()
    image = cv2.resize(image, (100, 100))
    hog = descriptor.HOG(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # When generating features
    features = hog.describe([image])

    # It should be true sample
    assert features.shape == (1L, 4356L)


if __name__ == "__main__":
    import nose
    nose.run()    
    
    
    
    
    
