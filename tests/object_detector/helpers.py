
import cv2
import numpy as np
import os

np.random.seed(1111)
    
def get_one_sample_image():
    from skimage import data
    image = data.camera()
    image = cv2.resize(image, (100, 100))
    return image


def create_empty_files(directory, files):
    os.mkdir(directory)
    files = [os.path.join(directory, afile) for afile in files]
    for filename in files:
        with open(filename, "w") as _: pass
    return files

