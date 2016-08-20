
import scanner
import descriptor
import classifier

class Detector(object):
    
    def __init__(self, descriptor, classifier):
        self.descriptor = descriptor
        self.classifier = classifier

    def run(self, image, window_size, step, pyramid_scale=0.7, threshold_prob=0.7):
        scanner_ = scanner.ImageScanner(image)
        
        boxes = []
        probs = []
        
        for _ in scanner_.get_next_layer(pyramid_scale, window_size):
            for _, _, window in scanner_.get_next_patch(step[0], step[1], window_size[0], window_size[1]):
                
                features = self.descriptor.describe([window]).reshape(1, -1)
                prob = self.classifier.predict_proba(features)[0][1]
                
                if prob > threshold_prob:
                    bb = scanner_.bounding_box
                    boxes.append(bb)
                    probs.append(prob)
        return boxes, probs
    
    def hard_negative_mine(self):
        pass
    
    
if __name__ == "__main__":
    
    desc = descriptor.HOG()
    cls = classifier.LinearSVM(C = 1.0, random_state = 111)

    #1. Get Features from training set
    
    #2. Training classifier and save them
    
    #3. Create detector and test
    import numpy as np
    sample_img = np.zeros((100, 100))
    detector = Detector(desc, cls)
    detector.run(sample_img)

    #4. Hard-Negative-Mine
    detector.hard_negative_mine()
    
    #5. Re-train classifier
    detector.classifier.train()

    #6. Test
    detector.run(sample_img)

    
    




