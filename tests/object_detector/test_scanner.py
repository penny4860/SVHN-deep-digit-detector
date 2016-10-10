
import cv2
import numpy as np
from object_detector.scanner import ImageScanner
import helpers

def test_build_image_pyramid():
    # Given one sample image (100,100) and the following parameters
    image = helpers.get_one_sample_image()
    parameters = {"scale": 0.5, "min_y": 20, "min_x": 20}
    
    # When building image pyramid
    scanner = ImageScanner(image)
    pyramid = [layer for layer in scanner.get_next_layer(scale=parameters["scale"], min_x=parameters["min_y"], min_y=parameters["min_x"])]
    
    # Then it requires the following condition
    # 1) number of pyramids 
    n_pyramids = 1
    layer_ = image
    while True:
        h = int(layer_.shape[0] * parameters['scale'])
        w = int(layer_.shape[1] * parameters['scale'])
        layer_ = cv2.resize(layer_, (w, h))
        if layer_.shape[0] < parameters['min_y'] and layer_.shape[1] < parameters['min_x']:
            break
        n_pyramids += 1
    assert len(pyramid) == n_pyramids, "ImageScanner.get_next_layer() unit test failed"

    # 2) similarity of image contents
    for layer in pyramid:
        img_from_layer = cv2.resize(layer, (image.shape[1], image.shape[0]))
        rel_error = np.mean(np.absolute(img_from_layer - image) / image)
        assert rel_error < np.max(image) * 0.03, "ImageScanner.get_next_layer() unit test failed. \
            Relative Error between original image and layer should be less than 3% of maximum intensity"
        
def test_sliding_window():
    # Given one sample image (100,100) and the following parameters
    image = helpers.get_one_sample_image()
    parameters = {"scale": 0.5, "min_x": 20, "min_y": 20, "step_y": 10, "step_x": 10, "win_y": 30, "win_x": 30}

    # When performing sliding window in multi-pyramid
    scanner = ImageScanner(image)
    for layer in scanner.get_next_layer(parameters['scale'], parameters['min_y'], parameters['min_x']):
        
        test_yx_pairs = []
        for y in range(0, layer.shape[0] - parameters['win_y'], parameters['step_y']):
            for x in range(0, layer.shape[1] - parameters['win_x'], parameters['step_x']):
                test_yx_pairs.append((y,x))
        
        for i, (y, x, patch) in enumerate(scanner.get_next_patch(parameters['step_y'], parameters['step_x'], parameters['win_y'], parameters['win_x'])):
            assert patch.all() == layer[y:y+parameters['step_y'], x:x+parameters['step_x']].all()
            assert test_yx_pairs[i][0] == y and test_yx_pairs[i][1] == x
            
            #todo : bounding-box in original image (scanner.bounding_box)
            

import pytest
if __name__ == '__main__':
    pytest.main([__file__])
    
    
    



    
    
    
    
    
    
    
    
    
    
    