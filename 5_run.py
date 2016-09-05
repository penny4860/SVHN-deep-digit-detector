
import object_detector.file_io as file_io
import object_detector.descriptor as descriptor
import object_detector.classifier as classifier
import object_detector.detector as detector
import cv2

CONFIGURATION_FILE = "conf/new_format.json"

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"], n_files_to_sample=10)

    # 2. Build detector and save it   
    hog = descriptor.HOG(conf["descriptor"]["parameters"]["orientations"],
                         conf["descriptor"]["parameters"]["pixels_per_cell"],
                         conf["descriptor"]["parameters"]["cells_per_block"])
    cls = classifier.LinearSVM.load(conf["classifier"]["output_file"])
    detector = detector.Detector(hog, cls)
    #detector.dump(conf["descriptor"]["output_file"])

    # 3. Run detector on Test images 
    for image_file in test_image_files:
        test_image = cv2.imread(image_file)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
      
        print "[INFO] Test Image shape: {0}".format(test_image.shape)

        boxes, probs = detector.run(test_image, 
                                    conf["detector"]["window_dim"], 
                                    conf["detector"]["window_step"], 
                                    conf["detector"]["pyramid_scale"], 
                                    conf["detector"]["min_probability"])
        detector.show_boxes(test_image, boxes)
    
    
        




