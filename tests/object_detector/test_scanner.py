
import time
import cv2
import object_detector.scanner as scanner
from skimage import data

if __name__ == "__main__":
    
    image = data.camera()        # Get Sample Image
    image = cv2.resize(image, (100, 100))
    image_scanner = scanner.ImageScanner(image)
    
    for layer in image_scanner.get_next_layer():
        for x, y, window in image_scanner.get_next_patch():
            clone = layer.copy()
            cv2.rectangle(clone, (x, y), (x + 30, y + 30), (0, 255, 0), 2)
            cv2.imshow("Test Image Scanner", clone)
            cv2.waitKey(1)
            time.sleep(0.025)

    
    

    
    