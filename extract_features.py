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
    getter = descriptor.FeatureGetter(descriptor=hog, patch_size=conf['window_dim'])
    
    # 3. Get Feature sets
    positive_dataset = getter.get_positive_sets(image_dir=conf["image_dataset"],
                                                        pattern="*.jpg", 
                                                        annotation_path=conf['image_annotations'],
                                                        padding=conf['offset'])
    negative_dataset = getter.get_negative_sets(image_files=conf["image_distractions"],
                                                        pattern="*.jpg",
                                                        n_samples_per_img=5,
                                                        sample_ratio=0.5)
    dataset = np.concatenate([positive_dataset, negative_dataset], axis=0)
    print("[INFO] number of positive data : {}, number of negative data : {}".format(positive_dataset.shape, negative_dataset.shape))

    # 4. Save dataset
    file_io.FileHDF5().write(dataset, "", "features")


    
