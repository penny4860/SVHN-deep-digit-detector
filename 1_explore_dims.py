#-*- coding: utf-8 -*-

""" Get information on training dataset."""

import object_detector.file_io as file_io
import object_detector.extractor as extractor

CONFIGURATION_FILE = "conf/car_side.json"

if __name__ == "__main__":
    conf_ = file_io.FileJson().read(CONFIGURATION_FILE)
    h, w = extractor.calc_average_patch_size(conf_["dataset"]["annotations_dir"], "*.mat")
    print "[INFO] average (height, width) = {:.2f}, {:.2f}".format(h, w)





    

