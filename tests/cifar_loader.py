
import cPickle
import numpy as np
import cv2

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

files = ['../../datasets/svhn/cifar-10-batches-py/data_batch_1']

dict = unpickle(files[0])

images = dict['data'].reshape(-1, 3, 32, 32)
labels = np.array(dict['labels'])
images = np.swapaxes(images, 1, 3)


#images[0] = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
cv2.imshow("", images[1000])
cv2.waitKey(0)
cv2.destroyAllWindows()

