#-*- coding: utf-8 -*-

# 1. MSER 로 Candidate Region 을 찾는다
# 2. Candidate Region 을 32x32x1 로 resize
#         (w >= h) : 32x32 로 rescale
#         (w < h) : w=h 가 되도록 crop 후 32x32 로 rescale
#         natural 영상의 edge 부에서의 처리 ?
#            - 균등하게 분할 할 수 있는 만큼만 crop 해서 rescale

img_files = ['imgs/1.png', 'imgs/2.png']

import cv2
import numpy as np

import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.utils as utils

def propose_patches(image, dst_size=(32, 32)):
    detector = rp.MserDetector()
    candidates_bbs = detector.detect(img, False)

    patches = []
    for bb in candidates_bbs:
        y1, y2, x1, x2 = bb
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        
        if width >= height:
            pad_y = 0
            pad_x = 0
        else:
            pad_x = int((height-width)/2)
            pad_y = 0
        sample = utils.crop_bb(img, bb, pad_size=(pad_y ,pad_x), dst_size=(32,32))
        patches.append(sample)
    return np.array(patches)

# 1. image files
img_file = img_files[0]

# 2. image
img = cv2.imread(img_file)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. region proposals (N, 32, 32, 3)
patches = propose_patches(img)
show.plot_images(patches)

# 4. Convert to gray
patches = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches]
patches = np.array(patches)
patches = patches.reshape(-1, 32, 32, 1)
print patches.shape

# 5. Run pre-trained classifier




