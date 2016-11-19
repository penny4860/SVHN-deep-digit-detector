#-*- coding: utf-8 -*-
import object_detector.file_io as file_io
import object_detector.factory as factory
import object_detector.evaluate as evaluate
import argparse as ap

DEFAULT_CONFIG_FILE = "conf/svhn.json"
DEFAULT_N_TEST_IMAGE = "all"                 

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-t', "--n_test_image", help="Number of Test Images", default=DEFAULT_N_TEST_IMAGE, type=str)
    args = vars(parser.parse_args())
    
    conf = file_io.FileJson().read(DEFAULT_CONFIG_FILE)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"])
    
    if args["n_test_image"] == "all":
        pass
    else:
        n_test_imgs = int(args["n_test_image"])
        test_image_files = test_image_files[:n_test_imgs]
    print len(test_image_files)

    # 2. Build detector
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], 
                                               conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], 
                                               conf["classifier"]["parameters"], 
                                               conf["classifier"]["output_file"])

    # 3. Evaluate average precision     
    evaluator = evaluate.Evaluator()
    ap = evaluator.eval_average_precision(test_image_files, 
                               conf["dataset"]["annotation_file"],
                               detector,
                               conf["detector"]["window_dim"],
                               conf["detector"]["window_step"],
                               conf["detector"]["pyramid_scale"],conf["detector"]["overlap_thresh"])
    print "Average Precision : {}".format(ap)
    evaluator.plot_recall_precision()
    # print evaluator.dataset
    
    
    
    
    
    
    
        




