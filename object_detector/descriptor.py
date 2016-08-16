#-*- coding: utf-8 -*-

import abc
import numpy as np
from skimage import feature

class Descriptor(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass

    @abc.abstractmethod    
    def describe(self, images, **kwargs):
        """
        images 
            list of image
            
        kwargs
        """

        pass

class HOG(Descriptor):
    
    def describe(self, images, **kwargs):
        orientations = kwargs.pop('orientations', False)
        pixels_per_cell = kwargs.pop('pixels_per_cell', False)
        cells_per_block = kwargs.pop('cells_per_block', False)
        
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r'.format(kwargs))
        
        features = []
        for image in images:
            feature_vector = feature.hog(image, 
                                   orientations=orientations, 
                                   pixels_per_cell=pixels_per_cell,
                                   cells_per_block=cells_per_block, 
                                   transform_sqrt=True)
            features.append(feature_vector)
            
        features = np.array(features)
        return features

class ColorHist(Descriptor):

    def describe(self):
        pass

class LBP(Descriptor):
    def __init__(self):
        pass

    def describe(self):
        pass

class HuMoments(Descriptor):
    def __init__(self):
        pass

    def describe(self):
        pass
    
if __name__ == "__main__":
    from skimage import data
    import cv2
    
    image = data.camera()        # Get Sample Image
    image = cv2.resize(image, (100, 100))

    hog = HOG()
    HH = hog.describe([image], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    # (4356L,), 0.0437659109109 0.0322149201473
    for H in HH:
        print H.shape, H[0], H[-1]








