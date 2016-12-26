#-*- coding: utf-8 -*-

import region_proposal as rp
import image_loader as loader
import cv2

# 1. file 을 load
files = loader.load_files(folder_name='../../datasets/svhn/train', n_images=2)

annotations = file_io.FileJson().read(annotation_file)


detector = rp.MserDetector()
for image_file in files:
    image = cv2.imread(image_file)
    candidates = detector.detect(image, True)
    
    
    # gts 를 구한다.
    


# 3. digitStruct.json file load



# 3. Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장


