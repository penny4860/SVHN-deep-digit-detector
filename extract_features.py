#-*- coding: utf-8 -*-

import numpy as np
import object_detector.file_io as file_io
import object_detector.descriptor as descriptor

if __name__ == "__main__":
    CONFIGURATION_FILE = "conf/cars.json"
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
     
    # 1. Initialize Descriptor instance
    hog = descriptor.HOG(conf['orientations'],
                         conf['pixels_per_cell'],
                         conf['cells_per_block'])
     
    # 2. Initialize FeatureGetter instance
    getter = file_io.FeatureGetter(descriptor=hog, patch_size=conf['window_dim'])
     
    # 3. Get Feature sets
    getter.add_positive_sets(image_dir=conf["image_dataset"],
                                                        pattern="*.jpg", 
                                                        annotation_path=conf['image_annotations'],
                                                        padding=conf['offset'])
    getter.add_negative_sets(image_dir=conf["image_distractions"],
                                                        pattern="*.jpg",
                                                        n_samples_per_img=5,
                                                        sample_ratio=0.5)
     
    getter.summary()
     
    # 4. Save dataset
    getter.save(config_file="feature_config.pkl", data_file="feature_data.hdf5")
    del getter
    
    getter = file_io.FeatureGetter.load(config_file="feature_config.pkl", data_file="feature_data.hdf5")
    getter.summary()


    
