
import object_detector.file_io as file_io
import object_detector.factory as factory
import cv2
import argparse as ap

DEFAULT_CONFIG_FILE = "conf/car_side.json"
DEFAULT_N_TEST_IMAGE = None                 # if this is None, it tests every images as possible
DEFAULT_NMS = True
DEFAULT_SHOW_OP = False

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-t', "--n_test_image", help="Number of Test Images", default=DEFAULT_N_TEST_IMAGE, type=int)
    parser.add_argument('-n', "--nms", help="Non Maxima Suppresiion", default=DEFAULT_NMS, type=bool)
    parser.add_argument('-s', "--show_operation", help="Show Detect Running Operation", default=DEFAULT_SHOW_OP, type=bool)
    args = vars(parser.parse_args())
    conf = file_io.FileJson().read(args["config"])

    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"])
    
    # Todo: Check this code
    if args["n_test_image"] is not None:
        test_image_files = test_image_files[:2]
    
    # 2. Build detector and save it   
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], 
                                               conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], 
                                               conf["classifier"]["parameters"], 
                                               conf["classifier"]["output_file"])

    # 3. Run detector on Test images 
    for image_file in test_image_files:
        test_image = cv2.imread(image_file)
        #test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        boxes, probs = detector.run(test_image, 
                                    conf["detector"]["window_dim"], 
                                    conf["detector"]["window_step"], 
                                    conf["detector"]["pyramid_scale"], 
                                    do_nms=args["nms"], 
                                    show_result=True,
                                    show_operation=args["show_operation"])
    
        




