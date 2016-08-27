
import object_detector.utils as utils

def test_crop_bb():
    # Given one sample image
    from skimage import data
    import cv2
    image = data.camera()
    image = cv2.resize(image, (100, 100))

    # When perform crop_bb()
    bb = (0, 10, 10, 20)    # y1, y2, x1, x2
    pad = 2
    desired_size = (6,6)
    patch = utils.crop_bb(image, bb, pad, desired_size)
    
    # Then it should be same with manually cropped one.
    crop_manual = image[max(bb[0],bb[0]-pad) : min(image.shape[0],bb[1]+pad), max(bb[2],bb[2]-pad) : min(image.shape[1],bb[3]+pad)]
    crop_manual = cv2.resize(crop_manual, desired_size, interpolation=cv2.INTER_AREA)
    assert patch.all() == crop_manual.all(), "utils.crop_bb() arises error!"

    
if __name__ == "__main__":
    import nose
    nose.run()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    