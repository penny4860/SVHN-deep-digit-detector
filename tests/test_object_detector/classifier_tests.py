#-*- coding: utf-8 -*-

import os
import numpy as np
import object_detector.classifier as classifier


def get_andgate_svm_classifier():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [0], [0], [1]]).reshape(-1, )
    cls = classifier.LinearSVM(C = 1.0, random_state = 111)
    cls.train(X, y)
    
    return cls

def test_predict_of_linear_svm():

    # Given And-Gate classifier
    cls = get_andgate_svm_classifier()
    
    # When predicting (2,2) sample
    y_pred = cls.predict([[2, 2]])

    # It should be true sample
    assert y_pred.all() == np.array([1]).all()
    
def test_predict_proba_of_linear_svm():

    # Given And-Gate classifier
    cls = get_andgate_svm_classifier()
    
    # When predicting (2,2) sample
    y_probs = cls.predict_proba([[2, 2]])
    probs_for_trues = y_probs[:, 0]
    
    # every probabilities for true should larger than 0.5
    assert probs_for_trues.any() > 0.5
    # sum of probabilities should be 1.0
    assert y_probs.sum(axis=1).any() == 1.0


def test_dump_and_load_of_linear_svm():

    # Given And-Gate classifier and test X-samples
    cls = get_andgate_svm_classifier()
    test_Xs = np.random.randn(10, 2)

    # When performing dump() and load() from another instance
    filename = "linear_svm.pkl"
    cls.dump(filename)
    cls_loaded = classifier.LinearSVM.load(filename)

    # the behavior between 2 instances should be same
    assert cls_loaded.predict(test_Xs).all() == cls.predict(test_Xs).all()
    assert cls_loaded.predict_proba(test_Xs).all() == cls.predict_proba(test_Xs).all()

    os.remove(filename)    
    
            
if __name__ == "__main__":
    import nose
    nose.run()    
    
    
    
    
    
    
