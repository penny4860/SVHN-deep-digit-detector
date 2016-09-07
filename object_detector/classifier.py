#-*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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

    @abstractmethod    
    def predict_proba(self, X):
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        return classification_report(y_test, self._model.predict(X_test))
    
    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._model, f)

#     @classmethod
#     def load(cls, filename):
#         with open(filename, 'rb') as f:
#             obj = pickle.load(f)
#         
#         loaded = cls(**obj['params'])
#         loaded._model = obj['model']
#         return loaded

class LinearSVM(Classifier):
    
    def __init__(self, C):
        self._C = C
        self._model = SVC(kernel="linear", C=C, probability=True)
        
    def train(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, X):
        return self._model.predict(X)
    
    def predict_proba(self, X):
        return self._model.predict_proba(X)
    

if __name__ == "__main__":
    pass
    
    
    

