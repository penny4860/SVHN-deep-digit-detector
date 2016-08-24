
import cv2
import numpy as np

import object_detector.file_io as file_io
import object_detector.descriptor as descriptor
import object_detector.classifier as classifier
import object_detector.detector as detector

CONFIGURATION_FILE = "conf/cars.json"

if __name__ == "__main__":
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    hog = descriptor.HOG(conf['orientations'],
                     conf['pixels_per_cell'],
                     conf['cells_per_block'])
    cls = classifier.LinearSVM.load(conf["classifier_path"])
  
    detector = detector.Detector(hog, cls)
    negative_image_files = file_io.list_files(conf["image_distractions"], "*.jpg", conf["hn_num_distraction_images"])
    features, probs = detector.hard_negative_mine(negative_image_files, 
                                               conf["window_dim"], 
                                               conf["window_step"], 
                                               conf["pyramid_scale"], 
                                               threshold_prob=0.5)
    
    negative_labels = np.zeros((len(features), 1))
    negative_set = np.concatenate([negative_labels, features], axis=1)
    file_io.FileHDF5().write(negative_set, conf["features_path"], "hard_negatives", write_mode="a")
    
    print len(negative_set)
    print "done"
