#-*- coding: utf-8 -*-
import os
import argparse as ap
import object_detector.file_io as file_io

DEFAULT_CONFIG_FILE = "conf/faces.json"

if __name__ == "__main__":
    
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    args = vars(parser.parse_args())
    
    conf = file_io.FileJson().read(args["config"])

    if os.path.exists(conf["dataset"]["pos_data_dir"]):
        print "Positive dataset location is correct"
    else:
        raise ValueError ("Positive dataset specified \"{0}\" is not exists. \n\
            Please check data path [\"dataset\"][\"pos_data_dir\"] in \"{0}\"".format(args["config"]))
     
    if os.path.exists(conf["dataset"]["neg_data_dir"]):
        print "Negative dataset location is correct"
    else:
        raise ValueError ("Negative dataset specified \"{0}\" is not exists. \n\
            Please check data path [\"dataset\"][\"neg_data_dir\"] in \"{0}\"".format(args["config"])) 



    

