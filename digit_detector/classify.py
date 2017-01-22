#-*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import keras

class Classifier:
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

    @abstractmethod
    def predict_proba(self, patches):
        pass
    
class CnnClassifier(Classifier):
    
    def __init__(self, model_file, input_shape=(32,32,1)):
        self._model = keras.models.load_model(model_file)
        self.input_shape = input_shape

    def predict_proba(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        probs = self._model.predict_proba(patches)
        return probs
    
class TrueBinaryClassifier(Classifier):
    """Classifier always predict true """
    def __init__(self, model_file=None, input_shape=None):
        self._model = None
        self.input_shape = None
    
    def predict_proba(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        probs = np.zeros((len(patches), 2))
        probs[probs[:, 1]] = 1
        
        return probs
