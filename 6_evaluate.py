#-*- coding: utf-8 -*-
import object_detector.file_io as file_io
import object_detector.factory as factory
import object_detector.evaluate as evaluate

CONFIGURATION_FILE = "conf/new_format.json"

if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    #test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"], n_files_to_sample=2)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"])
    test_image_files = test_image_files[:10]

    # 2. Build detector and save it   
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], conf["classifier"]["parameters"], conf["classifier"]["output_file"])
     
     
    ap = evaluate.Evaluator().eval_average_precision(test_image_files, 
                               conf["dataset"]['annotations_dir'], 
                               detector, 
                               conf["detector"]["window_dim"],
                               conf["detector"]["window_step"],
                               conf["detector"]["pyramid_scale"],
                               True)
    
    print ap
    
#     [INFO] Test Image shape: (197L, 300L)
#     [INFO] Test Image shape: (197L, 300L)
#     0.848484848485

    

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
        




