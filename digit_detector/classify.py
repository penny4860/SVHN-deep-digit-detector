#-*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import keras
import numpy as np

class Classifier:
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

    @abstractmethod
    def predict_proba(self, patches):
        pass
    
class CnnClassifier(Classifier):
    
    def __init__(self, model_file, preprocessor, input_shape=(32,32,1)):
        self._model = keras.models.load_model(model_file)
        self._preprocessor = preprocessor
        self.input_shape = input_shape

    def predict_proba(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        patches_preprocessed = self._preprocessor.run(patches)
        probs = self._model.predict_proba(patches_preprocessed, verbose=0)
        return probs
    
class TrueBinaryClassifier(Classifier):
    """Classifier always predict true """
    def __init__(self, model_file=None, preprocessor=None, input_shape=None):
        self._model = None
        self._preprocessor = None
        self.input_shape = input_shape
    
    def predict_proba(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        probs = np.zeros((len(patches), 2))
        probs[:, 1] = 1
        
        return probs
