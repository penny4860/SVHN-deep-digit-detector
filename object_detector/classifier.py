#-*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import pickle
import numpy as np
from sklearn.svm import SVC


class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod    
    def __init__(self, **params):
        raise NotImplementedError

    @abstractmethod    
    def train(self, X, y):
        raise NotImplementedError

    @abstractmethod    
    def predict(self, X):
        raise NotImplementedError
    
    def dump(self, filename):
        obj = {"model" : self._model, "params" : self._params}
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        
        loaded = cls(**obj['params'])
        loaded._model = obj['model']
        return loaded

class LinearSVM(Classifier):
    
    def __init__(self, **params):
        self._params = params
        self._model = SVC(kernel="linear", 
                          C=self._params['C'], 
                          probability=True, 
                          random_state=self._params['random_state'])
        
    def train(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, X):
        return self._model.predict(X)
    

if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
 
    obj = LinearSVM(C = 1.0, random_state = 111)
    obj.train(X, y)
    print obj.predict([[-0.8, -1]])
     
    obj.dump("linear_svm.pickle")
 
    obj2 = LinearSVM.load("linear_svm.pickle")
    print obj2.predict([[-0.8, -1]])

