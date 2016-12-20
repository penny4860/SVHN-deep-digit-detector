#-*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import cv2
import utils

files = utils.list_files('../../datasets/svhn/train/negative_images', pattern="*.png", n_files_to_sample=5000, recursive_option=False)

print len(files)

patch_size = [32, 32]
n_samples_per_img = 10

negative_images = []

for image_file in files:
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.shape[0] >= patch_size[0] and image.shape[1] >= patch_size[0]:
        patches = utils.crop_random(image, patch_size, n_samples_per_img)
        negative_images.append(patches)
    
negative_images = np.concatenate(negative_images, axis=0)
print negative_images.shape
# (39350, 32, 32)

cv2.imshow("", negative_images[-2])
cv2.waitKey()

