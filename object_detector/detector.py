#-*- coding: utf-8 -*-

import scanner
import numpy as np
import cv2
import pickle
import time


class Detector(object):
    
    def __init__(self, descriptor, classifier):
        self.descriptor = descriptor
        self.classifier = classifier
        
    def dump(self, filename):
        obj = {"descriptor" : self.descriptor, "classifier" : self.classifier}
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        
        loaded = cls(descriptor = obj["descriptor"], classifier = obj["classifier"])
        return loaded

    def run(self, image, window_size, step, pyramid_scale=0.7, threshold_prob=0.7, do_nms=True, show_operation=False):
        """
        
        Parameters
        ----------
        image : array, shape (n_rows, n_cols, n_channels) or (n_rows, n_cols)
            Input image to run the detector
            
        Returns
        ----------
        boxes : array, shape (n_detected, height, 4)
            detected bounding boxes 
        
        probs : array, shape (n_detected, 1)
            probability at the boxes
        """
        
        self._display_image = image
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image
        else:
            raise ValueError('Input image is invalid.')
            
        scanner_ = scanner.ImageScanner(gray_image)
        
        boxes = []
        probs = []
        
        # Todo : 모든 patch 를 generate 한 다음 한번에 연산하는 것과 속도를 비교해보자.
        for _ in scanner_.get_next_layer(pyramid_scale, window_size[0], window_size[1]):
            for _, _, window in scanner_.get_next_patch(step[0], step[1], window_size[0], window_size[1]):
                
                features = self.descriptor.describe([window]).reshape(1, -1)
                prob = self.classifier.predict_proba(features)[0][1]
                
                if prob > threshold_prob:
                    bb = scanner_.bounding_box
                    boxes.append(bb)
                    probs.append(prob)
                
                if show_operation:
                    if prob > threshold_prob:
                        pass
                    else:
                        pass
                    self.show_boxes([scanner_.bounding_box], 
                                    msg="{:.2f}".format(prob), 
                                    delay=0.02, 
                                    color=(0,255,0))

                        
        if do_nms and boxes != []:
            boxes, probs = self._do_nms(boxes, probs, overlapThresh=0.3)
            
        boxes = np.array(boxes, "int")
        probs = np.array(probs)
        return boxes, probs
    
    def show_boxes(self, boxes, msg=None, delay=None, color=(0,0,255)):
        
        image = self._display_image.copy()
        
        for y1, y2, x1, x2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if msg is not None:
                cv2.putText(image, msg, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

        cv2.imshow("Image", image)
        if delay is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
            time.sleep(0.025)
    
    def hard_negative_mine(self, negative_image_files, window_size, step, pyramid_scale=0.7, threshold_prob=0.5):

        # Todo : progress bar
        features = []
        probs = []
        
        for patch, probability in self._generate_negative_patches(negative_image_files, window_size, step, pyramid_scale, threshold_prob):
            
            feature = self.descriptor.describe([patch])[0]
            features.append(feature)
            probs.append(probability)
        
        if len(probs) == 0:
            pass
        
        else:
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
    pass
    




