
import object_detector.file_io as file_io
import object_detector.factory as factory
import cv2
import numpy as np
import pickle

CONFIGURATION_FILE = "conf/new_format.json"

def sort_by_probs(patches, probs):
    patches = np.array(patches)
    probs = np.array(probs)
    data = np.concatenate([probs.reshape(-1,1), patches], axis=1)
    data = data[data[:, 0].argsort()[::-1]]
    patches = data[:, 1:]
    probs = data[:, 0]
    return patches, probs


if __name__ == "__main__":
    
    # 1. Load configuration file and test images
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    test_image_files = file_io.list_files(conf["dataset"]["pos_data_dir"], n_files_to_sample=1)

    # 2. Build detector and save it   
    detector = factory.Factory.create_detector(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"],
                                               conf["classifier"]["algorithm"], conf["classifier"]["parameters"], conf["classifier"]["output_file"])
#     patches = []
#    
#     # 3. Run detector on Test images 
#     for image_file in test_image_files:
#         test_image = cv2.imread(image_file)
#         test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#           
#         print "[INFO] Test Image shape: {0}".format(test_image.shape)
#     
#         boxes, probs = detector.run(test_image, 
#                                     conf["detector"]["window_dim"], 
#                                     conf["detector"]["window_step"], 
#                                     conf["detector"]["pyramid_scale"], 
#                                     threshold_prob=0.0)
#         detector.show_boxes(test_image, boxes)
#         patches += boxes.tolist()
#         
#     # temporaty save boxes
#     with open("patches.pkl", 'wb') as f:
#         pickle.dump(patches, f)
#     with open("probs.pkl", 'wb') as f:
#         pickle.dump(probs.tolist(), f)


    # load boxes
    with open("patches.pkl", 'rb') as f:
        patches = pickle.load(f)
    with open("probs.pkl", 'rb') as f:
        probs = pickle.load(f)

    patches, probs = sort_by_probs(patches, probs)

    
    
    
    
    
    
    
    
    
    
    
    
        




