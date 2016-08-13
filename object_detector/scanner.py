#-*- coding: utf-8 -*-

class ImageScanner(object):
    def __init__(self, image):
        self._image = image
        self._layer = image
    
    def get_next_patch(self, step_size=(10, 10), window_size=(30, 30)):
        for y in range(0, self._layer.shape[0] - window_size[0], step_size[0]):
            for x in range(0, self._layer.shape[1] - window_size[1], step_size[1]):
                yield (x, y, self._layer[y:y + window_size[1], x:x + window_size[0]])
    
    def get_next_layer(self, scale=0.7, min_size=(30, 30)):
        layer = self._image
        self._layer = layer
        yield layer

        while True:
            h = int(layer.shape[0] * scale)
            w = int(layer.shape[1] * scale)
            layer = cv2.resize(layer, (w, h))
            
            min_h = min_size[0]
            min_w = min_size[1]
            if h < min_h or w < min_w:
                break
            self._layer = layer
            yield layer
    
    
if __name__ == "__main__":

    import cv2
    import time
    
    image = cv2.imread("test.jpg")
    scanner = ImageScanner(image)
    # loop over the layers of the image pyramid and display them
    #for layer in scanner.get_next_layer():
    layer = image
    
    print "image height = {}, image width = {}".format(image.shape[0], image.shape[1])
    for x, y, window in scanner.get_next_patch():
        clone = layer.copy()
        cv2.rectangle(clone, (x, y), (x + 30, y + 30), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        print x, x + 30, y, y + 30
    
        cv2.waitKey(1)
        time.sleep(0.025)

