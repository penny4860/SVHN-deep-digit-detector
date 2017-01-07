#-*- coding: utf-8 -*-
import cv2
import numpy as np

class Preprocessor:
    
    def __init__(self):
        pass
    
    def run(self):
        pass
    
class GrayImgPreprocessor(Preprocessor):
    def run(self, patches):
        """
        Parameters:
            patches (ndarray of shape (N, n_rows, n_cols, ch))
        Returns:
            patches (ndarray of shape (N, n_rows, n_cols, 1))
        """
        n_images, n_rows, n_cols, _ = patches.shape
        
        patches = np.array([self._to_gray(patch) for patch in patches], dtype='float')
        patches = patches.reshape(n_images, n_rows, n_cols, 1)
        return patches
    
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
    
    def _substract_mean(self, images, mean_value):
        """
        Parameters:
            images (ndarray of shape (N, n_rows, n_cols, ch))
            mean_vlaue (float)
        """
        images_zero_mean = images - mean_value
        return images_zero_mean

def to_gray(images):
    grays = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
    return np.array(grays)

def preprocess(images_train, labels_train, images_val, labels_val, nb_classes=2):
    from keras.utils import np_utils
    
    # 1. convert to gray
    X_train = to_gray(images_train).reshape(-1,32,32,1).astype('float32')
    X_val = to_gray(images_val).reshape(-1,32,32,1).astype('float32')

    y_train = labels_train.astype('int')
    y_val = labels_val.astype('int')
    y_train[y_train > 0] = 1
    y_val[y_val > 0] = 1

    mean_value = X_train.mean()
    
    X_train -= mean_value
    X_val -= mean_value

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
 
    return X_train, X_val, Y_train, Y_val, mean_value
