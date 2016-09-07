
import object_detector.file_io as file_io
import object_detector.factory as factory
import cv2

CONFIGURATION_FILE = "conf/new_format.json"

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"], n_files_to_sample=10)

    # 2. Build detector and save it   
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], conf["classifier"]["parameters"], conf["classifier"]["output_file"])

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
    
    
        




