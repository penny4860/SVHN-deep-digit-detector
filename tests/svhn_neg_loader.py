#-*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import cv2
import utils

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def load_images(folder_name='../../datasets/svhn/train', n_images=None):
    """
    Returns 
        images (list of image (n_rows, n_cols))
    """
    files = utils.list_files(folder_name, pattern="*.png", n_files_to_sample=None, recursive_option=False)
    sort_nicely(files)
    
    if n_images is not None:
        files = files[:n_images]
    
    images = []
    for image_file in files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
    return images

# 1. svhn natural image 를 모두 load
images = load_images(folder_name='../../datasets/svhn/train', n_images=2)

# 2. MSER 로 region detection
for img in images:
    mser = cv2.MSER(_delta = 1)
    regions = mser.detect(img, None)

    # loop over the contours
    for region in regions:
        clone = img.copy()
        # fit a bounding box to the contour
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        if h > w:
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow('img', clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



# 3. Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장


