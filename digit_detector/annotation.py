#-*- coding: utf-8 -*-

import file_io
import os
import numpy as np

def get_bbs(annotation):
    
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
        
def get_annotation(filename, annotation_file):
    annotations = file_io.FileJson().read(annotation_file)
    
    _, filename_ = os.path.split(filename)
    index = int(filename_[:filename_.rfind(".")])
    annotation = annotations[index-1]
    
    if annotation["filename"] == filename_:
        bbs, labels = get_bbs(annotation)
        return bbs, labels
        
    else:
        print "Annotation file should be sorted!!!!"






