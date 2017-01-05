#-*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class _Cropper:
    """ This class is an abstract class. """
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    
    def crop(self, image, box):
        """ Template Method """
        
        pad_x, pad_y = self._get_pad(image, box)
        
        h = image.shape[0]
        w = image.shape[1]
        (y1, y2, x1, x2) = box
        
        (x1, y1) = (max(x1 - pad_x, 0), max(y1 - pad_y, 0))
        (x2, y2) = (min(x2 + pad_x, w), min(y2 + pad_y, h))
        patch = image[y1:y2, x1:x2]
        return patch
    
    @abstractmethod    
    def _get_pad(self, image, box):
        pass

class CropperWithoutPad(_Cropper):
    
    def _get_pad(self, image, box):
        pad_x = 0
        pad_y = 0
        return pad_x, pad_y
    

class CropperWidthMargin(_Cropper):

    def _get_pad(self, image, box):
        height = image.shape[0]
        width = image.shape[1]
        
        if width >= height:
            pad_y = 0
            pad_x = 0
        else:
            pad_x = int((height-width)/2)
            pad_y = 0
        return pad_x, pad_y
