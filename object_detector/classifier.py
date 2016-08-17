#-*- coding: utf-8 -*-

import abc
import numpy as np
from skimage import feature

import cPickle

class Classifier(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod    
    def __init__(self, params):
        pass

    @abc.abstractmethod    
    def train(self, X, y):
        pass

    @abc.abstractmethod    
    def predict(self, X):
        pass

    @abc.abstractmethod    
    def dump(self, filename):
        obj = {"model" : self._model, "params" : self._params}
        cPickle.dumps(obj)

    @abc.abstractmethod    
    def load(self, filename):
        self._model = cPickle.loads(open(filename).read())
    

class LinearSVM(Classifier):
    
    def __init__(self, C, random_state=111):
        self._C = C
        self._random_state = random_state
        
    def train(self, X, y):
        self._model = SVC(kernel="linear", C=self._C, probability=True, random_state=self._random_state)
        self._model.fit(X, y)
    
    def predict(self, X):
        self._model.predict(X)
    

if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(X, y) 
    print(clf.predict([[-0.8, -1]]))







