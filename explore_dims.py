#-*- coding: utf-8 -*-

"""
Usage Example:
    python explore_dims --conf conf/cars.json
"""

import argparse
import numpy as np
import object_detector.file_io as file_io

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

conf_ = file_io.ReadJson().read(args['conf'])

widths = []
heights = []

files = file_io.list_files(conf_["image_annotations"], "*.mat")
for file_ in files:  
    (y, h, x, w) = file_io.ReadMat().read(file_)["box_coord"][0]
    widths.append(w - x)
    heights.append(h - y)
  
# compute the average of both the width and height lists
(avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
print("[INFO] avg. width: {:.2f}".format(avgWidth))
print("[INFO] avg. height: {:.2f}".format(avgHeight))
print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))


