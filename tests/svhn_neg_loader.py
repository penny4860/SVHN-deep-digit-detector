#-*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import cv2
import utils
from matplotlib import pyplot as plt

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
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
    return images

def detect_regions(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mser = cv2.MSER(_delta = 1)
    regions = mser.detect(gray, None)

    bbs = []
    for i, region in enumerate(regions):
        clone = img.copy()
        cv2.drawContours(clone, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        bbs.append((x, y, w, h))
    return regions, bbs

def plot_regions(img):
    
    regions, bbs = detect_regions(img)
    
    n_regions = len(regions)
    n_rows = int(np.sqrt(n_regions)) + 1
    n_cols = int(np.sqrt(n_regions)) + 2
    
    # plot original image 
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
      
    # plot edge map
    edges = cv2.Canny(img,10,20)
    plt.subplot(n_rows, n_cols, n_rows * n_cols)
    plt.imshow(edges, cmap = 'gray')
    plt.title('Edge Map'), plt.xticks([]), plt.yticks([])
      
    for i, region in enumerate(regions):
        clone = img.copy()
        cv2.drawContours(clone, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 1)
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
        plt.title('Contours'), plt.xticks([]), plt.yticks([])
     
    plt.show()

# 1. svhn natural image 를 모두 load
images = load_images(folder_name='../../datasets/svhn/train', n_images=2)

boxes = []

# 2. MSER 로 region detection
for img in images:
    plot_regions(img)
    _, bbs = detect_regions(img)
    boxes.append(bbs)

print len(boxes), len(boxes[0]), len(boxes[1])

# 3. Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장


