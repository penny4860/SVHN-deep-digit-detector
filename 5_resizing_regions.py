#-*- coding: utf-8 -*-
import cv2
import numpy as np

import digit_detector.region_proposal as rp
import digit_detector.show as show

def load_model(filename):
    from keras.models import load_model
    model = load_model(filename)
    return model

img_files = ['imgs/1.png', 'imgs/2.png']


if __name__ == "__main__":
    # 1. image files
    img_file = img_files[1]
    
    # 2. image
    img = cv2.imread(img_file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. region proposals (N, 32, 32, 3)
    patches, bbs = rp.propose_patches(img)
    temp = patches
    
    # 4. Convert to gray
    patches = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches]
    patches = np.array(patches)
    patches = patches.reshape(-1, 32, 32, 1)
    print patches.shape
    
    # 5. Run pre-trained classifier
    model = load_model("detector_model.hdf5")
    
    probs = model.predict_proba(patches)[:, 1]
    
    print probs.shape
     
    show.plot_images(temp, probs.tolist())
    show.plot_bounding_boxes(img, bbs, probs.tolist())




