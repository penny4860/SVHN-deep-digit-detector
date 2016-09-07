#-*- coding: utf-8 -*-

import abc
import numpy as np
from skimage import feature


class Descriptor(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod    
    def __init__(self, params):
        pass

    @abc.abstractmethod    
    def describe(self, images):
        pass


class HOG(Descriptor):
    
    def __init__(self, 
                 orientations=9, 
                 pixels_per_cell=(8, 8), 
                 cells_per_block=(2, 2)):
        self._orientations = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block
    
    def describe(self, images):
        features = []
        for image in images:
            feature_vector = feature.hog(image, 
                                   orientations=self._orientations, 
                                   pixels_per_cell=self._pixels_per_cell,
                                   cells_per_block=self._cells_per_block, 
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
    pass








