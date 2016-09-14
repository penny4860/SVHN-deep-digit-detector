#-*- coding: utf-8 -*-
import subprocess
import object_detector.file_io as file_io

CONFIG_FILE = "conf/car_side.json"
N_RUNNING_TEST_IMAGE = 5                 
N_EVALUATION_TEST_IMAGE = "all"

if __name__ == "__main__":
    
    conf = file_io.FileJson().read(CONFIG_FILE)
    
    print "\n[Step 0] Checking dataset path"
    subprocess.call("python 0_check_dataset_path.py -c {}".format(CONFIG_FILE))
    
    print "\n[Step 1] Getting average size of the target object.\n\
            You can set or edit your detector's window size referring this result.\n\
            config_file.json[\"detector\"][\"window_dim\"]\n\
            [INFO] current setting is {}".format(conf["detector"]["window_dim"])
            
    subprocess.call("python 1_get_object_size.py -c {}".format(CONFIG_FILE))
    
    print "\n[Step 2] Extracting features"
    subprocess.call("python 2_extract_features.py -c {}".format(CONFIG_FILE))
    
    print "\n[Step 3] Training classifier. (Note: It can be time-consuming task)"
    subprocess.call("python 3_train.py -c {} -i {}".format(CONFIG_FILE, False))
    
    print "\n[Step 4] Gathering Hard Negative Samples"
    subprocess.call("python 4_hard_negative_mine.py -c {}".format(CONFIG_FILE))
    
    print "\n[Step 5] Re-training classifier including hard-negative-mined samples. (Note: It can be time-consuming task)"
    subprocess.call("python 3_train.py -c {} -i {}".format(CONFIG_FILE, True))
    
    print "\n[Step 6] Running detector"
    subprocess.call("python 5_run.py -c {} -t {} -n {} -s {}".format(CONFIG_FILE, N_RUNNING_TEST_IMAGE, True, True))
    
    print "\n[Step 7] Evaluate detector using average-precision metric"
    subprocess.call("python 6_evaluate.py -c {} -t {}".format(CONFIG_FILE, N_EVALUATION_TEST_IMAGE))
    
    


