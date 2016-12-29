#-*- coding: utf-8 -*-

# 1. MSER 로 Candidate Region 을 찾는다
# 2. Candidate Region 을 32x32x1 로 resize
#         (w >= h) : 32x32 로 rescale
#         (w < h) : w=h 가 되도록 crop 후 32x32 로 rescale
#         natural 영상의 edge 부에서의 처리 ?
#            - 균등하게 분할 할 수 있는 만큼만 crop 해서 rescale

def load_model(filename):
    from keras.models import load_model
    model = load_model(filename)
#     score = model.evaluate(X_test, Y_test, verbose=0)
#     print('Test score:', score[0])
#     print('Test accuracy:', score[1])
    return model

img_files = ['imgs/1.png', 'imgs/2.png']

import cv2
import numpy as np

import digit_detector.region_proposal as rp
import digit_detector.show as show

# 1. image files
img_file = img_files[0]

# 2. image
img = cv2.imread(img_file)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. region proposals (N, 32, 32, 3)
patches = rp.propose_patches(img)
show.plot_images(patches)

# 4. Convert to gray
patches = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches]
patches = np.array(patches)
patches = patches.reshape(-1, 32, 32, 1)
print patches.shape

# 5. Run pre-trained classifier
model = load_model("models/detector_model.hdf5")
print "Done"

model.


