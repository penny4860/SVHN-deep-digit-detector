#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import object_detector.classifier as classifier
import numpy as np

CONFIGURATION_FILE = "conf/cars.json"
HARD_NEGATIVE_OPTION = False

if __name__ == "__main__":
    
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    
    #1. Load Features and Labels
    # Todo : feature 와 label 을 나누어서 hdf file 에 쓰자.
    data = file_io.FileHDF5().read(conf["features_path"], conf["db_name"])
    if HARD_NEGATIVE_OPTION:
        hard_negative_data = file_io.FileHDF5().read(conf["features_path"], "hard_negatives")
        data = np.concatenate([data, hard_negative_data], axis=0)

    y = data[:, 0]
    X = data[:, 1:]

    #2. Train
    cls = classifier.LinearSVM(C=conf["C"], random_state=111)
    cls.train(X, y)
    print "[INFO] Training result is as follows"
    print cls.evaluate(X, y)
    #              precision    recall  f1-score   support
    # 
    #         0.0       0.99      1.00      1.00      5000
    #         1.0       1.00      0.74      0.85       122
    # 
    # avg / total       0.99      0.99      0.99      5122
 
    #3. Save classifier
    cls.dump(conf["classifier_path"])


