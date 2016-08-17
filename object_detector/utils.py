
import cv2
 
def crop_bb(image, bb, padding=10, dstSize=(32, 32)):
    
    h = image.shape[0]
    w = image.shape[0]

    (y1, y2, x1, x2) = bb
    
    (x1, y1) = (max(x1 - padding, 0), max(y1 - padding, 0))
    (x2, y2) = (min(x2 + padding, w), min(y2 + padding, h))
    
    roi = image[y1:y2, x1:x2]
    roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)
 
    return roi


