#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import object_detector.factory as factory
import argparse as ap

DEFAULT_HNM_OPTION = True
DEFAULT_CONFIG_FILE = "conf/svhn.json"

if __name__ == "__main__":
    
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-i', "--include_hnm", help="Include Hard Negative Mined Set", default=DEFAULT_HNM_OPTION, type=bool)
    args = vars(parser.parse_args())
    
    conf = file_io.FileJson().read(args["config"])
    
    #1. Load Features and Labels
    getter = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], 
                                              conf["descriptor"]["parameters"], 
                                              conf["detector"]["window_dim"], 
                                              conf["extractor"]["output_file"])
    getter.summary()
    features, labels = getter.get_dataset(include_hard_negative=args["include_hnm"])
    
    print type(features), type(labels)
    print features.shape, labels.shape
    
    import keras
    
    
    
#     y = data[:, 0].astype(int)
#     y[y > 0] = 1
#     X = data[:, 1:]
#     
#     #2. Load classifier and Train
#     cls = factory.Factory.create_classifier(conf["classifier"]["algorithm"], 
#                                             conf["classifier"]["parameters"])
#     
#     cls.train(X, y)
#     print "[INFO] Training result is as follows"
#     print cls.evaluate(X, y)
#  
#     #3. Save classifier
#     cls.dump(conf["classifier"]["output_file"])


