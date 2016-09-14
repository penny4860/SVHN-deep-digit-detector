
import object_detector.file_io as file_io
import object_detector.detector as detector
import object_detector.factory as factory
import argparse as ap

DEFAULT_CONFIG_FILE = "conf/car_side.json"

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    args = vars(parser.parse_args())
    conf = file_io.FileJson().read(args["config"])
    
    #1. Create detector
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], 
                                               conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], 
                                               conf["classifier"]["parameters"], 
                                               conf["classifier"]["output_file"])
    #2. Load negative images
    negative_image_files = file_io.list_files(conf["dataset"]["neg_data_dir"], 
                                              conf["dataset"]["neg_format"], 
                                              n_files_to_sample=conf["hard_negative_mine"]["n_images"])
    
    #3. Get hard negative mined features
    features, probs = detector.hard_negative_mine(negative_image_files, 
                                                  conf["detector"]["window_dim"], 
                                                  conf["hard_negative_mine"]["window_step"], 
                                                  conf["hard_negative_mine"]["pyramid_scale"], 
                                                  threshold_prob=conf["hard_negative_mine"]["min_probability"])
    
    print "[HNM INFO] : number of mined negative patches {}".format(len(features))
    print "[HNM INFO] : probabilities of mined negative patches {}".format(probs)
    
    #4. Add hard negative mined features to the extractor
    extractor = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], 
                                              conf["descriptor"]["parameters"], 
                                              conf["detector"]["window_dim"], 
                                              conf["extractor"]["output_file"])
    print "Before adding hard-negative-mined samples"
    extractor.summary()
    extractor.add_data(features, -1)
    print "After adding hard-negative-mined samples"
    extractor.summary()
    
    #extractor.save(data_file=conf["extractor"]["output_file"])

    
    
    
    
    

