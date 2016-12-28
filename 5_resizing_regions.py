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


img_file = img_files[0]
img = cv2.imread(img_file)

detector = rp.MserDetector()
candidates_bbs = detector.detect(img, False)

samples = []

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
    samples.append(sample)

print len(samples)
show.plot_images(samples)






