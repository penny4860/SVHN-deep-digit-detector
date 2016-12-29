#-*- coding: utf-8 -*-
import cv2
import numpy as np
import keras.models

import digit_detector.region_proposal as rp
import digit_detector.show as show


def detect(image, model_filename, mean_value, input_shape = (32,32,1), threshold=0.9):
    patches, bbs = rp.propose_patches(image, dst_size = (input_shape[0], input_shape[1]))
#     temp = patches
    
    # 4. Convert to gray
    patches = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches]
    patches = np.array(patches)
    patches = patches.astype('float32')
    patches -= mean_value

    patches = patches.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    
    # 5. Run pre-trained classifier
    model = keras.models.load_model(model_filename)
    probs = model.predict_proba(patches)[:, 1]
     
#     show.plot_images(temp, probs.tolist())
#     show.plot_bounding_boxes(image, bbs, probs.tolist())

    bbs = bbs[probs > threshold]
    probs = probs[probs > threshold]
    
    for i, bb in enumerate(bbs):
        image = show.draw_box(image, bb, 2)
            
    cv2.imshow("MSER + CNN", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    pass






