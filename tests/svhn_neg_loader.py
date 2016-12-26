#-*- coding: utf-8 -*-

import region_proposal as rp
import file_io
import image_loader as loader
import cv2
import os

import matplotlib.pyplot as plt

N_IMAGES = 2
DIR = '../../datasets/svhn/train'


def get_bbs(annotation):
    
    bbs = []
    labels = []
    
    for box in annotation["boxes"]:
        x1 = int(box["left"])
        y1 = int(box["top"])
        w = int(box["width"])
        h = int(box["height"])

        bb = (y1, y1+h, x1, x1+w)
        label = int(box["label"])
        
        bbs.append(bb)
        labels.append(label)
    return bbs, labels
        
def get_annotation(filename, annotation_file):
    annotations = file_io.FileJson().read(annotation_file)
    
    _, filename_ = os.path.split(filename)
    index = int(filename_[:filename_.rfind(".")])
    annotation = annotations[index-1]
    
    if annotation["filename"] == filename_:
        bbs, labels = get_bbs(annotation)
        return bbs, labels
        
    else:
        print "Annotation file should be sorted!!!!"

# 1. file 을 load
# Todo : refactoring : load_files 함수를 file_io 모듈에 넣자. 
files = loader.load_files(folder_name='../../datasets/svhn/train', n_images=N_IMAGES)
annotation_file = "../../datasets/svhn/train/digitStruct.json"
detector = rp.MserDetector()


for image_file in files:
    image = cv2.imread(image_file)
    candidates = detector.detect(image, True)
    
    gts, labels = get_annotation(image_file, annotation_file)
    
    for ground_truth in gts:
        y1, y2, x1, x2 = ground_truth
        
        clone = image.copy()
        cv2.rectangle(clone, (x1, y1), (x2, y2), (255, 0, 0), 1)
    plt.imshow(clone)
    plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    

    
    


# 3. digitStruct.json file load



# 3. Ground Truth 와의 overlap 이 5% 미만인 모든 sample 을 negative set 으로 저장


