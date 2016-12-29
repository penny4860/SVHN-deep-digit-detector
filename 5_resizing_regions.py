#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.region_proposal as rp
import digit_detector.show as show
import digit_detector.cnn as cnn


img_files = ['imgs/1.png', 'imgs/2.png']
model_filename = "detector_model.hdf5"
mean_value = 109.467

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
    patches = patches.astype('float32')
    patches -= mean_value

    patches = patches.reshape(-1, 32, 32, 1)
    print patches.shape
    
    # 5. Run pre-trained classifier
    model = keras.models.load_model(model_filename)
    
    probs = model.predict_proba(patches)[:, 1]
    
    print probs.shape
     
    show.plot_images(temp, probs.tolist())
    show.plot_bounding_boxes(img, bbs, probs.tolist())




