
import object_detector.utils as utils
import cv2
import numpy as np
np.random.seed(1111)

    
def get_one_sample_image():
    from skimage import data
    image = data.camera()
    image = cv2.resize(image, (100, 100))
    return image

def test_crop_bb():
    # Given one sample image and the following parameters
    image = get_one_sample_image()
    parameters = {"bb" : (0, 10, 10, 20),
        "pad" : 2,
        "desired_size" : (6,6),
    }
    
    # When perform crop_bb()
    patch = utils.crop_bb(image, bb=parameters["bb"], padding=parameters["pad"], dst_size=parameters["desired_size"])
    
    # Then it should be same with manually cropped one.
    bb = parameters["bb"]
    pad = parameters["pad"]
    desired_size = parameters["desired_size"]
    crop_manual = image[max(bb[0],bb[0]-pad) : min(image.shape[0],bb[1]+pad), max(bb[2],bb[2]-pad) : min(image.shape[1],bb[3]+pad)]
    crop_manual = cv2.resize(crop_manual, desired_size, interpolation=cv2.INTER_AREA)
    assert patch.all() == crop_manual.all(), "utils.crop_bb() arises error!"

def test_crop_random():
    # Given one sample image and the following parameters
    image = get_one_sample_image()
    parameters = {"dst_size" : (20, 20),
        "n_patches" : 5,
    }
    
    # When perform crop_random()
    patches = utils.crop_random(image, parameters["dst_size"], parameters["n_patches"])

    # Then every patch should be included in an image.
    match_cost = []
    for patch in patches:
        M = cv2.matchTemplate(image, patch, cv2.TM_SQDIFF)
        min_cost, _, _, _ = cv2.minMaxLoc(M)
        match_cost.append(min_cost)
    assert np.array(match_cost).all() == 0

if __name__ == "__main__":
    import nose
    nose.run()    
    

    
    
    



    
    
    
    
    
    
    
    
    
    
    