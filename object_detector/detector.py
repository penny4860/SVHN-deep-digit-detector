
import scanner
import descriptor
import classifier

import numpy as np

class Detector(object):
    
    def __init__(self, descriptor, classifier):
        self.descriptor = descriptor
        self.classifier = classifier

    def run(self, image, window_size, step, pyramid_scale=0.7, threshold_prob=0.7, do_nms=True):
        scanner_ = scanner.ImageScanner(image)
        
        boxes = []
        probs = []
        
        for _ in scanner_.get_next_layer(pyramid_scale, window_size[0], window_size[1]):
            for _, _, window in scanner_.get_next_patch(step[0], step[1], window_size[0], window_size[1]):
                
                features = self.descriptor.describe([window]).reshape(1, -1)
                prob = self.classifier.predict_proba(features)[0][1]
                
                if prob > threshold_prob:
                    bb = scanner_.bounding_box
                    boxes.append(bb)
                    probs.append(prob)
        
        if do_nms:
            boxes, probs = self._do_nms(boxes, probs, overlapThresh=0.3)
            
        boxes = np.array(boxes, "int")
        probs = np.array(probs)
        return boxes, probs
    
    def show_boxes(self, image, boxes):
        for y1, y2, x1, x2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    
    def hard_negative_mine(self, negative_image_files, window_size, step, pyramid_scale=0.7, threshold_prob=0.5):

        features = []
        probs = []
        
        for patch, probability in self._generate_negative_patches(negative_image_files, window_size, step, pyramid_scale, threshold_prob):
            
            feature = self.descriptor.describe([patch])[0]
            features.append(feature)
            probs.append(probability)
        
        # sort by probability        
        data = np.concatenate([np.array(probs).reshape(-1,1), np.array(features)], axis=1)
        data = data[data[:, 0].argsort()[::-1]]
        features = data[:, 1:]
        probs = data[:, 0]
        
        return features, probs

    def _generate_negative_patches(self, negative_image_files, window_size, step, pyramid_scale, threshold_prob):
        for image_file in negative_image_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
            # detect objects in the image
            (boxes, probs) = self.run(image, window_size, step, pyramid_scale, threshold_prob, do_nms=False)

            for (y1, y2, x1, x2), prob in zip(boxes, probs):
                negative_patch = cv2.resize(image[y1:y2, x1:x2], (window_size[1], window_size[0]), interpolation=cv2.INTER_AREA)
                yield negative_patch, prob

    # todo: code review
    def _do_nms(self, boxes, probs, overlapThresh=0.5):
        if len(boxes) == 0:
            return []
     
        boxes = np.array(boxes, dtype="float")
        probs = np.array(probs)
     
        pick = []
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]
     
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(probs)
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value to the list of
            # picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            # find the largest (x, y) coordinates for the start of the bounding box and the
            # smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
    
            # delete all indexes from the index list that have overlap greater than the
            # provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        # return only the bounding boxes that were picked
        return boxes[pick].astype("int"), probs[pick]
    
if __name__ == "__main__":
    import file_io
    import cv2
    import random
    test_image_file = "C:/datasets/caltech101/101_ObjectCategories/car_side/image_0002.jpg"
    test_image = cv2.imread(test_image_file)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
     
    print "[INFO] Test Image shape: {0}".format(test_image.shape)
    CONFIGURATION_FILE = "../conf/cars.json"
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
  
    hog = descriptor.HOG(conf['orientations'],
                     conf['pixels_per_cell'],
                     conf['cells_per_block'])
    cls = classifier.LinearSVM.load(conf["classifier_path"])
  
    detector = Detector(hog, cls)
#     boxes, probs = detector.run(test_image, conf["window_dim"], conf["window_step"], conf["pyramid_scale"], conf["min_probability"])
#     detector.show_boxes(test_image, boxes)
 
    negative_image_files = file_io.list_files(conf["image_distractions"], "*.jpg")
    negative_image_files = random.sample(negative_image_files, conf["hn_num_distraction_images"])
    #negative_image_files = random.sample(negative_image_files, 1)
    
    #4. Hard-Negative-Mine
    features, probs = detector.hard_negative_mine(negative_image_files, 
                                               conf["window_dim"], 
                                               conf["window_step"], 
                                               conf["pyramid_scale"], 
                                               threshold_prob=0.01)
     
#     #5. Re-train classifier
#     detector.classifier.train(X=features, y=)

    #6. Test
    #detector.run(test_image)

    
    




