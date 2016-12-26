#-*- coding: utf-8 -*-

import region_proposal as rp
import image_loader as loader

# 1. svhn natural image 를 모두 load
images = loader.load_images(folder_name='../../datasets/svhn/train', n_images=2)

boxes = []

# 2. MSER 로 region detection
detector = rp.MserDetector()

for img in images:
    bounding_boxes = detector.detect(img, True)
    boxes.append(bounding_boxes)

print len(boxes), len(boxes[0]), len(boxes[1])

# 3. digitStruct.json file load



# 3. Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장


