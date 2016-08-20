
import scanner
import descriptor
import classifier

class Detector(object):
    
    def __init__(self, descriptor, classifier):
        self.descriptor = descriptor
        self.classifier = classifier

    def run(self, image):
        image_scanner = scanner.ImageScanner(image)
    
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

    
    




