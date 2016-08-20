
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
        
        for _ in scanner_.get_next_layer(pyramid_scale, window_size[0], window_size[1]):
            for _, _, window in scanner_.get_next_patch(step[0], step[1], window_size[0], window_size[1]):
                
                features = self.descriptor.describe([window]).reshape(1, -1)
                prob = self.classifier.predict_proba(features)[0][1]
                
                if prob > threshold_prob:
                    bb = scanner_.bounding_box
                    boxes.append(bb)
                    probs.append(prob)
        return boxes, probs
    
    def show_boxes(self, image, boxes):
        for y1, y2, x1, x2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    
    def hard_negative_mine(self):
        pass
    
if __name__ == "__main__":
    import file_io
    import cv2
    test_image_file = "C:/datasets/caltech101/101_ObjectCategories/car_side/image_0025.jpg"
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
    boxes, probs = detector.run(test_image, conf["window_dim"], conf["window_step"], conf["pyramid_scale"], conf["min_probability"])
    detector.show_boxes(test_image, boxes)
 
#     #4. Hard-Negative-Mine
#     detector.hard_negative_mine()
#      
#     #5. Re-train classifier
#     detector.classifier.train()

    #6. Test
    #detector.run(test_image)

    
    




