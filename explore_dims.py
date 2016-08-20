#-*- coding: utf-8 -*-

""" Get information on training dataset."""

import numpy as np
import object_detector.file_io as file_io

CONFIGURATION_FILE = "conf/cars.json"

if __name__ == "__main__":
    conf_ = file_io.FileJson().read(CONFIGURATION_FILE)
    
    widths = []
    heights = []
    
    files = file_io.list_files(conf_["image_annotations"], "*.mat")
    for file_ in files:  
        (y, h, x, w) = file_io.FileMat().read(file_)["box_coord"][0]
        widths.append(w - x)
        heights.append(h - y)
      
    # compute the average of both the width and height lists
    (avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
    print("[INFO] avg. width: {:.2f}".format(avgWidth))
    print("[INFO] avg. height: {:.2f}".format(avgHeight))
    print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))
    

