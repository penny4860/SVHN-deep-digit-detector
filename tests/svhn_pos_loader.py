import numpy as np
import scipy.io as io
import cv2

def load_svhn_images(file_name):
    dataset = io.loadmat(file_name)
    images = dataset['X']
    labels = dataset['y']
    images = np.transpose(images, (3, 0, 1, 2))
    
    # (N, rows, cols, channels)
    # (N, 1)
    return images, labels

images, labels = load_svhn_images('../../datasets/svhn/train_32x32.mat')    
print images.shape, labels.shape

# cv2.imshow("", images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

