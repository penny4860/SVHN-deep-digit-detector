#-*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from keras.utils import np_utils

import cv2
import numpy as np


class _Preprocessor:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

    def _to_gray(self, image):
        """
        Parameters:
            image (ndarray of shape (n_rows, n_cols, ch) or (n_rows, n_cols))
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image
        else:
            raise ValueError("image dimension is strange")
        return gray_image


class _TrainTimePreprocessor(_Preprocessor):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    @abstractmethod
    def run(self, images_train, labels_train, images_val, labels_val, nb_classes=2):
        pass


class GrayImgTrainPreprocessor(_TrainTimePreprocessor):
    
    def __init__(self):
        pass

    def run(self, images_train, labels_train, images_val, labels_val, nb_classes=2):
        
        _, n_rows, n_cols, ch = images_train.shape
        
        # 1. convert to gray images
        X_train = np.array([self._to_gray(patch) for patch in images_train], dtype='float').reshape(-1, n_rows, n_cols, 1)
        X_val = np.array([self._to_gray(patch) for patch in images_val], dtype='float').reshape(-1, n_rows, n_cols, 1)
        
        # convert class vectors to binary class matrices
        y_train = labels_train.astype('int')
        y_val = labels_val.astype('int')

        if nb_classes == 2:
            y_train[y_train > 0] = 1
            y_val[y_val > 0] = 1
        elif nb_classes == 10:
            X_train = X_train[y_train[:,0] > 0, :, :, :]
            X_val = X_val[y_val[:,0] > 0, :, :, :]
            y_train = y_train[y_train > 0]
            y_val = y_val[y_val > 0]
            y_train[y_train == 10] = 0
            y_val[y_val == 10] = 0
            
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_val = np_utils.to_categorical(y_val, nb_classes)

        # 2. calc mean value
        mean_value = X_train.mean()
        X_train -= mean_value
        X_val -= mean_value
     
        return X_train, X_val, Y_train, Y_val, mean_value


class _RunTimePreprocessor(_Preprocessor):
    __metaclass__ = ABCMeta
    
    def __init__(self, mean_value=None):
        self._mean_value = mean_value
    
    @abstractmethod
    def run(self, patches):
        pass
    
    def _substract_mean(self, images):
        """
        Parameters:
            images (ndarray of shape (N, n_rows, n_cols, ch))
            mean_vlaue (float)
        """
        images_zero_mean = images - self._mean_value
        return images_zero_mean
    
class GrayImgPreprocessor(_RunTimePreprocessor):
    def run(self, patches):
        """
        Parameters:
            patches (ndarray of shape (N, n_rows, n_cols, ch))
        Returns:
            patches (ndarray of shape (N, n_rows, n_cols, 1))
        """
        n_images, n_rows, n_cols, _ = patches.shape
        
        patches = np.array([self._to_gray(patch) for patch in patches], dtype='float')
        patches = self._substract_mean(patches)
        patches = patches.reshape(n_images, n_rows, n_cols, 1)
        return patches
    
class NonePreprocessor(_RunTimePreprocessor):
    def run(self, patches):
        return patches
    
    
    
