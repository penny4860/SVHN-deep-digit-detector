
import cv2
import numpy as np

import object_detector.file_io as file_io
import object_detector.descriptor as descriptor
import object_detector.classifier as classifier
import object_detector.detector as detector
import object_detector.extractor as extractor


CONFIGURATION_FILE = "conf/new_format.json"
PATCH_SIZE = (32, 96)

if __name__ == "__main__":
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    
    desc = descriptor.DescriptorFactory.create(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"])
    cls = classifier.LinearSVM.load(conf["classifier"]["output_file"])
    
    detector = detector.Detector(desc, cls)
    negative_image_files = file_io.list_files(conf["dataset"]["neg_data_dir"], 
                                              conf["dataset"]["neg_format"], 
                                              n_files_to_sample=conf["hard_negative_mine"]["n_images"])
    
    features, probs = detector.hard_negative_mine(negative_image_files, 
                                               conf["detector"]["window_dim"], 
                                               conf["hard_negative_mine"]["window_step"], 
                                               conf["hard_negative_mine"]["pyramid_scale"], 
                                               threshold_prob=conf["hard_negative_mine"]["min_probability"])
    
    print "[HNM INFO] : number of mined negative patches {}".format(len(features))
    print "[HNM INFO] : probabilities of mined negative patches {}".format(probs)
    
    getter = extractor.FeatureExtractor.load(descriptor=desc, patch_size=PATCH_SIZE, data_file=conf["extractor"]["output_file"])
    getter.summary()
    getter.add_data(features, -1)
    getter.summary()
    
    getter.save(data_file=conf["extractor"]["output_file"])

    
    
    
    
    

