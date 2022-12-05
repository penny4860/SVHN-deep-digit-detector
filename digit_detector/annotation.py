#-*- coding: utf-8 -*-
from digit_detector import file_io
import os
import numpy as np

class Annotation:
    
    def __init__(self, annotation_file):
        self._load_annotation_file(annotation_file)
    
class SvhnAnnotation(Annotation):
    
    def get_boxes_and_labels(self, image_file):
        
        annotation = self._get_annotation(image_file)
        
        bbs = []
        labels = []
        
        for box in annotation["boxes"]:
            x1 = int(box["left"])
            y1 = int(box["top"])
            w = int(box["width"])
            h = int(box["height"])
    
            bb = (y1, y1+h, x1, x1+w)
            label = int(box["label"])
            
            bbs.append(bb)
            labels.append(label)
        return np.array(bbs), np.array(labels)
            
    def _load_annotation_file(self, annotation_file):
        self._annotations = file_io.FileJson().read(annotation_file)
    
    def _get_annotation(self, image_file):
        
        _, image_file = os.path.split(image_file)
        index = int(image_file[:image_file.rfind(".")])
        annotation = self._annotations[index-1]

        if annotation["filename"] != image_file:
            raise ValueError("Annotation file should be sorted!!!!")
        else:
            return annotation


        

