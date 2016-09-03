
import object_detector.file_io as file_io
import object_detector.descriptor as descriptor
import object_detector.classifier as classifier
import object_detector.detector as detector
import cv2

CONFIGURATION_FILE = "conf/cars.json"

if __name__ == "__main__":

    #import progressbar
    test_image_file = "C:/datasets/caltech101/101_ObjectCategories/car_side/image_0010.jpg"
    test_image = cv2.imread(test_image_file)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
      
    print "[INFO] Test Image shape: {0}".format(test_image.shape)
    conf = file_io.FileJson().read(CONFIGURATION_FILE)

    # 1. Build detector and save it   
    hog = descriptor.HOG(conf['orientations'],
                     conf['pixels_per_cell'],
                     conf['cells_per_block'])
    cls = classifier.LinearSVM.load(conf["classifier_path"])
   
    detector = detector.Detector(hog, cls)
    boxes, probs = detector.run(test_image, conf["window_dim"], conf["window_step"], conf["pyramid_scale"], conf["min_probability"])
    detector.show_boxes(test_image, boxes)
    detector.dump("detector_car.pkl")
    
    # 2. load saved classifier
    d = detector.Detector.load("detector_car.pkl")
    boxes, probs = d.run(test_image, conf["window_dim"], conf["window_step"], conf["pyramid_scale"], conf["min_probability"])
    d.show_boxes(test_image, boxes)




