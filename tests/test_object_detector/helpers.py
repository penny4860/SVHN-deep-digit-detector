
import cv2
import numpy as np
np.random.seed(1111)
    
def get_one_sample_image():
    from skimage import data
    image = data.camera()
    image = cv2.resize(image, (100, 100))
    return image


