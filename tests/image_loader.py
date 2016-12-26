#-*- coding: utf-8 -*-

import cv2
import utils
import re

class FileSorter:
    def __init__(self):
        pass
    
    def sort(self, list_of_strs):
        list_of_strs.sort(key=self._alphanum_key)

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s
    
    def _alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ self._tryint(c) for c in re.split('([0-9]+)', s) ]

def load_files(folder_name='../../datasets/svhn/train', n_images=None):
    files = utils.list_files(folder_name, pattern="*.png", n_files_to_sample=None, recursive_option=False)
    FileSorter().sort(files)
    if n_images is not None:
        files = files[:n_images]
    return files

def load_images(files, n_images=None):
    """
    Returns 
        images (list of image (n_rows, n_cols))
    """
    
    images = []
    for image_file in files:
        image = cv2.imread(image_file)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
    return images
