#-*- coding: utf-8 -*-
import descriptor
import classifier


class Factory(object):
     
    def __init__(self):
        pass
     
    @staticmethod
    def create_descriptor(desc_type, params):
         
        valid_types = descriptor.Descriptor.__subclasses__()
        valid_types_name = [type_.__name__ for type_ in valid_types]
        
        assert desc_type in valid_types_name, "Bad creation. Check desc_type parameter"
         
        if desc_type == "HOG":
            return descriptor.HOG(**params)
#         if desc_type == "LBP":
#             return descriptor.LBP(**params)
  
 
#     @staticmethod
#     def create_classifier(self, type_, params):
#         
#         if type_ == "LinearSVM":
#             desc = classifier.LinearSVM(**params)
#             
#         #Todo :type check detector �� subclass name
#         return desc
    @staticmethod
    def create_detector(self, desc_type, desc_param, cls_type, cls_param, cls_file):
        pass

Factory.create_descriptor("HOG", params=None)
