#-*- coding: utf-8 -*-

import cv2
import utils
import object_detector.file_io as file_io
import numpy as np
import random

class FeatureExtractor():
    
    def __init__(self, descriptor, patch_size, data_file):
        self._desc = descriptor
        self._patch_size = patch_size
        
        if data_file is None:
            self._features = None   # (N, n, m, ...)     
            self._labels = None     # (N, 1)
        else:
            self._features = file_io.FileHDF5().read(data_file, "features")
            self._labels = file_io.FileHDF5().read(data_file, "labels")
    
    def add_positive_sets(self, annotation_file, sample_ratio=1.0, padding=0, label=1):
        
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
        
        # Todo : list 를 모두 ndarray 로 바꾸자.
        self._features = np.concatenate([self._features, np.array(features_set)], axis = 0) if self._features is not None else np.array(features_set)
        self._labels = np.concatenate([self._labels, np.array(labels_set).reshape(-1, 1)], axis = 0) if self._labels is not None else np.array(labels_set).reshape(-1, 1)


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

        # Todo : list 를 모두 ndarray 로 바꾸자.
        self._features = np.concatenate([self._features, np.array(features_set)], axis = 0) if self._features is not None else np.array(features_set)
        self._labels = np.concatenate([self._labels, np.zeros((len(features_set), 1))], axis = 0) if self._labels is not None else np.zeros((len(features_set), 1))
        
        
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


import os
def calc_average_patch_size(annotation_file):
    widths = []
    heights = []
    
    annotations = file_io.FileJson().read(annotation_file)
    """    
    {
        u'boxes': 
            [
                {u'height': 219.0, u'width': 81.0, u'top': 77.0, u'left': 246.0, u'label': 1.0}, 
                {u'height': 219.0, u'width': 96.0, u'top': 81.0, u'left': 323.0, u'label': 9.0}
            ], 
        u'filename': u'1.png'
    }
    """
    image_path = os.path.split(annotation_file)[0]
    
    for annotation in annotations:
        image = cv2.imread(os.path.join(image_path, annotation["filename"]))

        for box in annotation["boxes"]:
            x1 = int(box["left"])
            y1 = int(box["top"])
            w = int(box["width"])
            h = int(box["height"])
            label = int(box["label"])
            cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
            cv2.putText(image, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

            widths.append(w)
            heights.append(h)

    height = np.mean(heights)
    width = np.mean(widths)
    
    print "h_min", np.min(heights)
    print "h_max", np.max(heights)
    print "w_min", np.min(widths)
    print "w_max", np.max(widths)
    
    data = np.array([(h, w, h+w) for h, w in zip(heights, widths)])
    print data.shape
    
    data = data[data[:, 0].argsort()]
    print data[:20]

    data = data[data[:, 1].argsort()]
    print data[:20]

    data = data[data[:, 2].argsort()]
    print data[:20]
    return height, width


