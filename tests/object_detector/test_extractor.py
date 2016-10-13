#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import os
import object_detector.descriptor as desc
import object_detector.extractor as extractor_


def setup_extractor(datafile = None):
    descriptor = desc.HOG(9, [4,4], [2,2])
    extractor = extractor_.FeatureExtractor(descriptor, [32, 16], datafile)
    return extractor 

def setup_params():
    annotation_filename = "../datasets/positive/digitStruct.json"
    negative_dir = "../datasets/negative"
    output_file = "svhn_features.hdf5"
    return annotation_filename, negative_dir, output_file

def test_add_dataset_behavior():
    
    extractor = setup_extractor()
    annotation_filename, negative_dir, output_file = setup_params()

    # 2. Get Feature sets
    
    extractor.add_positive_sets(annotation_file=annotation_filename,
                             sample_ratio=1.0,
                             padding=0,
                             )
    
    extractor.add_negative_sets(image_dir=negative_dir,
                             pattern="*.jpg",
                             n_samples_per_img=10,
                             sample_ratio=1.0)

    features, labels = extractor.get_dataset(include_hard_negative=True)
    
    n_positive_data = 8
    n_negative_data = 40
    assert features.shape == (n_positive_data + n_negative_data, 756)
    assert labels.shape == (n_positive_data + n_negative_data, 1)
    assert len(features[labels > 0]) == n_positive_data
    assert len(features[labels <= 0]) == n_negative_data

def test_save_and_load_extractor():
    extractor = setup_extractor()
    annotation_filename, negative_dir, output_file = setup_params()
      
    extractor.add_positive_sets(annotation_file=annotation_filename,
                             sample_ratio=1.0,
                             padding=0,
                             )
    extractor.add_negative_sets(image_dir=negative_dir,
                             pattern="*.jpg",
                             n_samples_per_img=10,
                             sample_ratio=1.0)
    
    # 3. Save dataset
    extractor.save(data_file=output_file)
    
    features, labels = extractor.get_dataset()
    features_loaded = file_io.FileHDF5().read(output_file, "features")
    labels_loaded = file_io.FileHDF5().read(output_file, "labels")
    
    assert features.all() == features_loaded.all()
    assert labels.all() == labels_loaded.all()
    
    extractor_loaded = setup_extractor(output_file)
    features_loaded, labels_loaded = extractor_loaded.get_dataset()

    assert features.all() == features_loaded.all()
    assert labels.all() == labels_loaded.all()
    os.remove(output_file)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
    





