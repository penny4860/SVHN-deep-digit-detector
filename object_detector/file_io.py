#-*- coding: utf-8 -*-

import abc
import glob
import os
import commentjson as json
from scipy import io
import numpy as np
import h5py
import random
import cv2
import utils
import pickle


class File(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def read(self, filename):
        pass
    
    @abc.abstractmethod
    def write(self, data, filename, write_mode="w"):
        pass
    

class FileJson(File):
    def read(self, filename):
        """load json file as dict object

        Parameters
        ----------
        filename : str
            filename of json file
    
        Returns
        ----------
        conf : dict
            dictionary containing contents of json file
    
        Examples
        --------
        """
        return json.loads(open(filename).read())
    
    # Todo : Exception 처리
    def write(self, data, filename, write_mode="w"):
        with open(filename, write_mode) as f:
            json.dump(data, f, indent=4)


class FileMat(File):
    def read(self, filename):
        """load mat file as dict object

        Parameters
        ----------
        filename : str
            filename of json file
    
        Returns
        ----------
        conf : dict
            dictionary containing contents of mat file
    
        Examples
        --------
        """
        return io.loadmat(filename)
    
    def write(self, data, filename, write_mode="w"):
        io.savemat(filename, data)


# Todo : staticmethod??
class FileHDF5(File):
    def read(self, filename, db_name):
        db = h5py.File(filename, "r")
        np_data = np.array(db[db_name])
        db.close()
        
        return np_data
    
    def write(self, data, filename, db_name, write_mode="w"):
        """Write data to hdf5 format.
        
        Parameters
        ----------
        data : array
            data to write
            
        filename : str
            filename including path
            
        db_name : str
            database name
            
        """
        
        # todo : overwrite check
        db = h5py.File(filename, write_mode)
        dataset = db.create_dataset(db_name, data.shape, dtype="float")
        dataset[:] = data[:]
        db.close()


class FeatureGetter():
    
    def __init__(self, descriptor, patch_size, dataset=[]):
        self._desc = descriptor
        self._patch_size = patch_size
        self._dataset = dataset
        
    def add_positive_sets(self, image_dir, pattern, annotation_path, sample_ratio=1.0, padding=5, augment=True, label=1):
        
        features_set = []
        image_files = self._get_image_files(image_dir, pattern, sample_ratio)
    
        for image_file in image_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_id = utils.get_file_id(image_file)
            
            annotation_file = "{}/annotation_{}.mat".format(annotation_path, image_id)
            bb = FileMat().read(annotation_file)["box_coord"][0]
            roi = utils.crop_bb(image, bb, padding=padding, dst_size=self._patch_size)
            
            patches = (roi, cv2.flip(roi, 1)) if augment else (roi,)
            
            # Todo : augment modulization
            features = self._desc.describe(patches)
            features_set += features.tolist()
            
        labels = np.zeros((len(features_set), 1)) + label
        dataset = np.concatenate([labels, np.array(features_set)], axis=1)
        self._dataset += dataset.tolist()


    def add_negative_sets(self, image_dir, pattern, n_samples_per_img, sample_ratio=1.0):
        
        features_set = []
        image_files = self._get_image_files(image_dir, pattern, sample_ratio)

        for image_file in image_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             
            patches = utils.crop_random(image, self._patch_size, n_samples_per_img)
             
            features = self._desc.describe(patches)
            features_set += features.tolist()

        labels = np.zeros((len(features_set), 1))
        dataset = np.concatenate([labels, np.array(features_set)], axis=1)
        self._dataset += dataset.tolist()
    
    def save(self, config_file, data_file):
        FileHDF5().write(np.array(self._dataset), data_file, "label_and_features")
        
        config = {"descriptor" : self._desc, "patch_size" : self._patch_size}
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)

    def summary(self):
        
        labels = np.array(self._dataset)[:, 0]
        feature_shape = np.array(self._dataset)[:, 1:].shape
        
        n_positive_samples = len(labels[labels > 0])
        n_negative_samples = len(labels[labels == 0])
                                 
        print "[FeatureGetter INFO] Positive samples: {}, Negative samples: {}".format(n_positive_samples, n_negative_samples)
        print "[FeatureGetter INFO] Feature Dimension: {}".format(feature_shape[1])

    @classmethod
    def load(cls, config_file, data_file=None):
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
        
        if data_file is None:
            dataset = []
        else:
            dataset = FileHDF5().read(data_file, "label_and_features")

        loaded = cls(descriptor=config["descriptor"], patch_size=config["patch_size"], dataset=dataset.tolist())
        return loaded

    @property
    def dataset(self):
        if self._dataset is None:
            raise ValueError('There is no dataset in this instance')
        else:
            return self._dataset
    
    def _get_image_files(self, directory, pattern, sample_ratio):
        image_files = list_files(directory, pattern)
        image_files = random.sample(image_files, int(len(image_files) * sample_ratio))
        return image_files


def list_files(directory, pattern="*.*", n_files_to_sample=None, recursive_option=True):
    """list files in a directory matched in defined pattern.

    Parameters
    ----------
    directory : str
        filename of json file

    pattern : str
        regular expression for file matching
    
    n_files_to_sample : int or None
        number of files to sample randomly and return.
        If this parameter is None, function returns every files.
    
    recursive_option : boolean
        option for searching subdirectories. If this option is True, 
        function searches all subdirectories recursively.
        
    Returns
    ----------
    conf : dict
        dictionary containing contents of json file

    Examples
    --------
    """

    if recursive_option == True:
        dirs = [path for path, _, _ in os.walk(directory)]
    else:
        dirs = [directory]
    
    files = []
    for dir_ in dirs:
        for p in glob.glob(os.path.join(dir_, pattern)):
            files.append(p)
    
    if n_files_to_sample is not None:
        files = random.sample(files, n_files_to_sample)

    return files

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    


