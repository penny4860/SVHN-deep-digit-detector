import numpy as np
import scipy.io as io
import cv2

files = ['../../datasets/svhn/train_32x32.mat']

dataset = io.loadmat(files[0])

images = dataset['X']
labels = dataset['y']

images = np.transpose(images, (3, 0, 1, 2))

print images.shape, labels.shape

# (N, rows, cols, channels)

cv2.imshow("", images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

