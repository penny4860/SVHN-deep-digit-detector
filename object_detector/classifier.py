#-*- coding: utf-8 -*-

import abc
import numpy as np
from skimage import feature
from sklearn.svm import SVC

import pickle

class Classifier(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod    
    def __init__(self, **params):
        raise NotImplementedError

    @abc.abstractmethod    
    def train(self, X, y):
        raise NotImplementedError

    @abc.abstractmethod    
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
        
        instance = cls(obj['params'])
        instance._model = obj['model']
        return instance

class LinearSVM(Classifier):
    
    def __init__(self, params):
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
    clf = SVC()
    clf.fit(X, y) 
    print(clf.predict([[-0.8, -1]]))
 
    params = {'C' : 1.0, 'random_state' : None}
 
    obj = LinearSVM(params)
    obj.train(X, y)
    print obj.predict([[-0.8, -1]])
     
    obj.dump("linear_svm.pickle")
 
    obj2 = LinearSVM.load("linear_svm.pickle")
    print obj2.predict([[-0.8, -1]])

