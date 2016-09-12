
import object_detector.file_io as file_io
import object_detector.factory as factory
import cv2
import numpy as np

CONFIGURATION_FILE = "conf/car_side.json"

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"])
    test_image_files = test_image_files[:1]
    
    # 2. Build detector and save it   
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], conf["classifier"]["parameters"], conf["classifier"]["output_file"])

    # 3. Run detector on Test images 
    for image_file in test_image_files:
        test_image = cv2.imread(image_file)
        #test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
      
        print "[INFO] Test Image shape: {0}".format(gray.shape)

        boxes, probs = detector.run(gray, 
                                    conf["detector"]["window_dim"], 
                                    conf["detector"]["window_step"], 
                                    conf["detector"]["pyramid_scale"], 
                                    0.5)
        detector.show_boxes(test_image, boxes)
        
        print type(boxes)
        print np.array(boxes).shape
        
        cv2.putText(test_image, 'car_side={:.2f}'.format(probs[0]), (boxes[0][2], boxes[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        cv2.imshow("Image", test_image)
        cv2.waitKey(0)

        #cv2.putText(test_image, 'Car : {}'.format(probs[0]), (0, 0), font, 6, (200,255,155), 13, cv2.LINE_AA)
    
    
        




