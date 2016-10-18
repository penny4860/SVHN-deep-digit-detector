#-*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sklearn
from sklearn import linear_model

class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod    
    def __init__(self, **params):
        raise NotImplementedError

    def train(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        return classification_report(y_test, self._model.predict(X_test))
    
    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._model, f)


class LinearSVM(Classifier):
    
    def __init__(self, C):
        self._C = C
        self._model = SVC(kernel="linear", C=C, probability=True)
        
    
class LogisticRegression(Classifier):
    
    def __init__(self, C):
        self._C = C
        self._model = linear_model.LogisticRegression(C=C)
        

class ConvNet(Classifier):
    
    def __init__(self, model_file):
        from keras.models import load_model
        self._model = load_model(model_file)

    def train(self):
        pass
    
    def dump(self):
        pass
    
    def evaluate(self):
        pass

if __name__ == "__main__":
    pass
    
    
    

