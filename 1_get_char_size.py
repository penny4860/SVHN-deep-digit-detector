#-*- coding: utf-8 -*-

""" Get information on training dataset."""

import object_detector.file_io as file_io
import numpy as np
import cv2
import os
import argparse as ap

DEFAULT_CONFIG_FILE = "conf/svhn.json"

def calc_average_patch_size(annotation_file):
    widths = []
    heights = []
    
    annotations = file_io.FileJson().read(annotation_file)
    """    
    {
        u'boxes': 
            [
                {u'height': 219.0, u'width': 81.0, u'top': 77.0, u'left': 246.0, u'label': 1.0}, 
                {u'height': 219.0, u'width': 96.0, u'top': 81.0, u'left': 323.0, u'label': 9.0}
            ], 
        u'filename': u'1.png'
    }
    """
    image_path = "../datasets/svhn/train"
    
    for annotation in annotations:
        image = cv2.imread(os.path.join(image_path, annotation["filename"]))

        for box in annotation["boxes"]:
            x1 = int(box["left"])
            y1 = int(box["top"])
            w = int(box["width"])
            h = int(box["height"])
            label = int(box["label"])
            cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
            cv2.putText(image, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

            widths.append(w)
            heights.append(h)

    height = np.mean(heights)
    width = np.mean(widths)
    
    print "h_min", np.min(heights)
    print "h_max", np.max(heights)
    print "w_min", np.min(widths)
    print "w_max", np.max(widths)
    
    data = np.array([(h, w, h+w) for h, w in zip(heights, widths)])
    print data.shape
    
    # w 가 최소인 pair, h 가 최소인 pair
    
    # 1. pair 의 1st dim 으로 정렬
    # 2. pair 의 2nd dim 으로 정렬
    # 3. pair 의 3rd dim 으로 정렬

    data = data[data[:, 0].argsort()]
    print data[:20]

    data = data[data[:, 1].argsort()]
    print data[:20]

    data = data[data[:, 2].argsort()]
    print data[:20]
    return height, width

if __name__ == "__main__":
    
    
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    args = vars(parser.parse_args())
    
    conf_ = file_io.FileJson().read(args["config"])
    h, w = calc_average_patch_size(conf_["dataset"]["annotation_file"])
    print "[INFO] average (height, width) = {:.2f}, {:.2f}".format(h, w)
    # 33.861310182 16.6504907381
    # (32, 16)
    
    
#     """    
#     {
#         u'boxes': 
#             [
#                 {u'height': 219.0, u'width': 81.0, u'top': 77.0, u'left': 246.0, u'label': 1.0}, 
#                 {u'height': 219.0, u'width': 96.0, u'top': 81.0, u'left': 323.0, u'label': 9.0}
#             ], 
#         u'filename': u'1.png'
#     }
#     """
#     image_path = "../datasets/svhn/train"
#     image = cv2.imread(os.path.join(image_path, annotations[0]["filename"]))
#     print image.shape
# 
#     annotations = file_io.FileJson().read("../datasets/svhn/train/digitStruct.json")
#     for box in annotations[0]["boxes"]:
#         x1 = int(box["left"])
#         y1 = int(box["top"])
#         w = int(box["width"])
#         h = int(box["height"])
#         label = int(box["label"])
#         cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
#         cv2.putText(image, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
#     cv2.imshow("Test", image)
#     cv2.waitKey(0)








    

