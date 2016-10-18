#-*- coding: utf-8 -*-
import descriptor
import classifier
import extractor
import detector

import pickle


class Factory(object):
     
    def __init__(self):
        raise NotImplementedError
     
    @staticmethod
    def create_descriptor(desc_type, params):
         
        valid_types = descriptor.Descriptor.__subclasses__()
        valid_types_name = [type_.__name__ for type_ in valid_types]
        
        assert desc_type in valid_types_name, "Bad creation. Check desc_type parameter"
         
        if desc_type == "HOG":
            return descriptor.HOG(**params)
        elif desc_type == "Image":
            return descriptor.Image(**params)
        
        assert 0, "Bad creation: " + desc_type
 
    @staticmethod
    def create_classifier(cls_type, params, model_file=None):

        valid_types = classifier.Classifier.__subclasses__()
        valid_types_name = [type_.__name__ for type_ in valid_types]
        assert cls_type in valid_types_name, "Bad creation. Check cls_type parameter"
        
        cls = None
         
        if cls_type == "LinearSVM":
            cls = classifier.LinearSVM(**params)
        elif cls_type == "LogisticRegression":
            cls = classifier.LogisticRegression(**params)
        elif cls_type == "ConvNet":
            cls = classifier.ConvNet(**params)
        
        if model_file is not None and cls_type != "ConvNet":
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                cls._model = model
        
        assert cls is not None, "Bad creation: " + cls_type
        return cls

    @staticmethod
    def create_extractor(desc_type, desc_param, patch_size, data_file=None):
        desc = Factory.create_descriptor(desc_type, desc_param)
        ext = extractor.FeatureExtractor(desc, patch_size, data_file)
        return ext
    
    @staticmethod
    def create_detector(desc_type, desc_param, cls_type, cls_param, model_file):
        desc = Factory.create_descriptor(desc_type, desc_param)
        cls = Factory.create_classifier(cls_type, cls_param, model_file)
        d = detector.Detector(desc, cls)
        return d

if __name__ == "__main__":
    
    parameters = {
            "orientations": 9,
            "pixels_per_cell": [4, 4],
            "cells_per_block": [2, 2]
        }
    
    desc = Factory.create_descriptor("HOG", params=parameters)
    print desc._cells_per_block

    parameters = {
            "C": 0.01
        }

    cls = Factory.create_classifier("LinearSVM", parameters)

    print cls._model

    parameters = {
            "model_file": "../models/detector_model.hdf5",
            "mean_value": 125.0,
        }
    cls = Factory.create_classifier("ConvNet", parameters)
    print cls._model










