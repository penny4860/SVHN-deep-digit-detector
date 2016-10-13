#-*- coding: utf-8 -*-

import cv2
import object_detector.utils as utils
import object_detector.file_io as file_io
import numpy as np
import random

import os

class FeatureExtractor():
    
    def __init__(self, descriptor, patch_size, data_file):
        self._desc = descriptor
        self._patch_size = patch_size
        
        if data_file is None:
            self._features = None   # (N, n, m, ...)     
            self._labels = None     # (N, 1)
#         else:
#             self._images = None
#             self._labels = None
#             
#             # self._dataset = file_io.FileHDF5().read(data_file, "label_and_features").tolist()
    
    # Todo : Template Method Pattern??
    def add_positive_sets(self, image_dir, pattern, annotation_path, sample_ratio=1.0, padding=5, augment=True, label=1):
        
        features_set = []
        image_files = self._get_image_files(image_dir, pattern, sample_ratio)
    
        for image_file in image_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_id = utils.get_file_id(image_file)
            
            annotation_file = "{}/annotation_{}.mat".format(annotation_path, image_id)
            bb = file_io.FileMat().read(annotation_file)["box_coord"][0]
            roi = utils.crop_bb(image, bb, padding=padding, dst_size=self._patch_size)
            
            patches = (roi, cv2.flip(roi, 1)) if augment else (roi,)
            
            # Todo : augment modulization
            features = self._desc.describe(patches)
            features_set += features.tolist()
            
        self._labels = np.zeros((len(features_set), 1)) + label
        self._features = np.array(features_set)


    def add_negative_sets(self, image_dir, pattern, n_samples_per_img, sample_ratio=1.0):
        # Todo : progressbar
        # Todo : 한꺼번에 write 하지말고, 10개 이미지를 모아서 write 를 자주하는 방식으로 수정
        
        features_set = []
        image_files = self._get_image_files(image_dir, pattern, sample_ratio)

        for image_file in image_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             
            patches = utils.crop_random(image, self._patch_size, n_samples_per_img)
             
            features = self._desc.describe(patches)
            features_set += features.tolist()

        self._labels = np.zeros((len(features_set), 1))
        self._features = np.array(features_set)
        
    def add_data(self, features, label):
        labels = np.zeros((len(features), 1)) + label
        
        if self._features is None:
            self._features = features
            self._labels = label
        else:
            self._features = np.concatenate([self._features, features], axis=0)
            self._labels = np.concatenate([self._labels, labels], axis=0)
        
    
    def save(self, data_file):
        file_io.FileHDF5().write(self._features, data_file, "features")
        file_io.FileHDF5().write(self._labels, data_file, "labels")


    def summary(self):
        
        labels = self._labels
        feature_shape = self._features.shape
        
        n_positive_samples = len(labels[labels > 0])
        n_negative_samples = len(labels[labels == 0])
        n_hard_negative_samples = len(labels[labels == -1])
                                 
        print "[FeatureGetter INFO] Positive samples: {}, Negative samples: {}, Hard Negative Mined samples: {}".format(n_positive_samples, n_negative_samples, n_hard_negative_samples)
        print "[FeatureGetter INFO] Feature Dimension: {}".format(feature_shape[1])


    def get_dataset(self, include_hard_negative=True):
        
        if self._features is None:
            raise ValueError('There is no dataset in this instance')
        else:
            labels = self._labels
            if include_hard_negative:
                features = self._features
                labels[labels < 0] = 0
            else:
                features = self._features[self._labels >= 0]
                labels = self._labels[self._labels >= 0]
            return features, labels
    
    
    def _get_image_files(self, directory, pattern, sample_ratio):
        image_files = file_io.list_files(directory, pattern)
        image_files = random.sample(image_files, int(len(image_files) * sample_ratio))
        return image_files


class SVHNFeatureExtractor(FeatureExtractor):
    # Todo : Template Method Pattern??
    def add_positive_sets(self, annotation_file, sample_ratio=1.0, padding=5, augment=True, label=1):
        
        features_set = []
        labels_set = []
        image_path = os.path.split(annotation_file)[0]
        annotations = file_io.FileJson().read(annotation_file)
        annotations = annotations[:int(len(annotations)*sample_ratio)]

        for annotation in annotations:
            image = cv2.imread(os.path.join(image_path, annotation["filename"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
            
            patches = []
            labels = []
            for box in annotation["boxes"]:
                x1 = int(box["left"])
                y1 = int(box["top"])
                w = int(box["width"])
                h = int(box["height"])

                bb = (y1, y1+h, x1, x1+w)
                label = int(box["label"])
                
                roi = utils.crop_bb(image, bb, padding=padding, dst_size=self._patch_size)
                patches.append(roi)
                labels.append(label)

            # Todo : augment modulization
            features = self._desc.describe(patches)
            features_set += features.tolist()
            labels_set += labels
            
        self._features = np.array(features_set)
        self._labels = np.array(labels_set).reshape(-1, 1)

import object_detector.descriptor as desc

def setup_extractor():
    descriptor = desc.HOG(9, [4,4], [2,2])
    extractor = SVHNFeatureExtractor(descriptor, [32, 16], None)
    return extractor 

def setup_params():
    annotation_filename = "../datasets/positive/digitStruct.json"
    negative_dir = "../datasets/negative"
    output_file = "svhn_features.hdf5"
    return annotation_filename, negative_dir, output_file

def test_add_positive_behavior():
    
    extractor = setup_extractor()
    annotation_filename, negative_dir, output_file = setup_params()

    # 2. Get Feature sets
    extractor.add_positive_sets(annotation_file=annotation_filename,
                             sample_ratio=1.0,
                             padding=0,
                             )

    features, labels = extractor.get_dataset(include_hard_negative=True)
    assert features.shape == (8, 756)
    assert labels.shape == (8, 1)
    

def test_add_negative_behavior():

    extractor = setup_extractor()
    annotation_filename, negative_dir, output_file = setup_params()
      
    # Todo : positive sample 숫자에 따라 negative sample 숫자를 자동으로 정할 수 있도록 설정
    extractor.add_negative_sets(image_dir=negative_dir,
                             pattern="*.jpg",
                             n_samples_per_img=10,
                             sample_ratio=1.0)
    
    features, labels = extractor.get_dataset(include_hard_negative=True)
    assert features.shape == (40, 756)
    assert labels.shape == (40, 1)

#     # 3. Save dataset
#     extractor.save(data_file=output_file)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
    





