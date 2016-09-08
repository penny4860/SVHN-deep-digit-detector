#-*- coding: utf-8 -*-
import object_detector.file_io as file_io
import object_detector.factory as factory
import cv2
import numpy as np
import pickle

CONFIGURATION_FILE = "conf/new_format.json"

def sort_by_probs(patches, probs):
    patches = np.array(patches)
    probs = np.array(probs)
    data = np.concatenate([probs.reshape(-1,1), patches], axis=1)
    data = data[data[:, 0].argsort()[::-1]]
    patches = data[:, 1:]
    probs = data[:, 0]
    return patches, probs

import object_detector.utils as utils

# Todo : extractor module 에 내용과 중복된다. 리팩토링하자.
def get_truth_bb(image_file, annotation_path):
    image_id = utils.get_file_id(image_file)
    annotation_file = "{}/annotation_{}.mat".format(annotation_path, image_id)
    bb = file_io.FileMat().read(annotation_file)["box_coord"][0]
    return bb

# intersection of union
def calc_iou(boxes, truth_box):
    y1 = boxes[:, 0]
    y2 = boxes[:, 1]
    x1 = boxes[:, 2]
    x2 = boxes[:, 3]
    
    y1_gt = truth_box[0]
    y2_gt = truth_box[1]
    x1_gt = truth_box[2]
    x2_gt = truth_box[3]
    
    xx1 = np.maximum(x1, x1_gt)
    yy1 = np.maximum(y1, y1_gt)
    xx2 = np.minimum(x2, x2_gt)
    yy2 = np.minimum(y2, y2_gt)

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    
    intersections = w*h
    As = (x2 - x1 + 1) * (y2 - y1 + 1)
    B = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    
    ious = intersections.astype(float) / (As + B -intersections)
    return ious

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"], n_files_to_sample=1)

    # 2. Build detector and save it   
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], conf["classifier"]["parameters"], conf["classifier"]["output_file"])
    patches = []
     
    # 3. Run detector on Test images 
    for image_file in test_image_files:
        test_image = cv2.imread(image_file)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            
        print "[INFO] Test Image shape: {0}".format(test_image.shape)
      
        boxes, probs = detector.run(test_image, 
                                    conf["detector"]["window_dim"], 
                                    conf["detector"]["window_step"], 
                                    conf["detector"]["pyramid_scale"], 
                                    threshold_prob=0.0)
         
        # Test Image �� ���� Ground-Truth �� Read
        truth_bb = get_truth_bb(image_file, conf["dataset"]['annotations_dir'])
        calc_iou(boxes, truth_bb)
         
        detector.show_boxes(test_image, boxes)
        patches += boxes.tolist()
          
    # temporaty save boxes
    with open("patches.pkl", 'wb') as f:
        pickle.dump(patches, f)
    with open("probs.pkl", 'wb') as f:
        pickle.dump(probs.tolist(), f)
    with open("gt.pkl", 'wb') as f:
        pickle.dump(truth_bb.tolist(), f)
    
#     # load boxes
#     with open("patches.pkl", 'rb') as f:
#         boxes = pickle.load(f)
#     with open("probs.pkl", 'rb') as f:
#         probs = pickle.load(f)
#     with open("gt.pkl", 'wb') as f:
#         truth_bb = pickle.load(f)
#  
#     ious = calc_iou(boxes, truth_bb)
#     print ious


    """
    2. Ground-Truth Box
    
        
    3. Plot the precision-recall graph
    
    4. Calculate interpolated precision on n-recall values
        * In the case of VOG challenge, n-values are evenly spaced 11 points

    5. Average interpolated precision values
        * It is Average Precision

    6. Repeat 1 ~ 6 for m-object and calculate mean of the average precision.
        * It is mAP
    """
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
        




