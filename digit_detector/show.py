#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Todo : 그림 그리는 방식을 데코레이터 패턴이로 정리해보자.

def draw_contour(image, region):
    image_drawn = image.copy()
    cv2.drawContours(image_drawn, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
    return image_drawn
    
def draw_box(image, box, thickness=4):
    image_drawn = image.copy()
    y1, y2, x1, x2 = box
    cv2.rectangle(image_drawn, (x1, y1), (x2, y2), (255, 0, 0), thickness)
    return image_drawn


def plot_contours(img, regions):
    n_regions = len(regions)
    n_rows = int(np.sqrt(n_regions)) + 1
    n_cols = int(np.sqrt(n_regions)) + 2
    
    # plot original image 
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
      
    for i, region in enumerate(regions):
        clone = img.copy()
        clone = draw_contour(clone, region)
        
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        clone = draw_box(clone, (y, y+h, x, x+w))
        
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
        plt.title('Contours'), plt.xticks([]), plt.yticks([])
     
    plt.show()


def plot_bounding_boxes(img, bounding_boxes, titles=None):
    """
    Parameters:
        img (ndarray)
        
        bounding_boxes (list of ndarray) : (y1, y2, x1, x2) ordered
    """
    
    n_regions = len(bounding_boxes)
    n_rows = int(np.sqrt(n_regions)) + 1
    n_cols = int(np.sqrt(n_regions)) + 2
    
    # plot original image 
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
    # Todo : plt 에서는 RGB 순으로 plot 한다. (opencv 에서는 BGR순) opencv 기준으로 정리하자.
    for i, box in enumerate(bounding_boxes):
        clone = img.copy()
        clone = draw_box(clone, box)
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
        if titles:
            plt.title("{0:.2f}".format(titles[i])), plt.xticks([]), plt.yticks([])
     
    plt.show()


def plot_images(images, titles=None):
    """
    Parameters:
        images (ndarray)
        
        titles (list of str)
    """
    n_images = len(images)
    n_rows = int(np.sqrt(n_images)) + 1
    n_cols = int(np.sqrt(n_images)) + 2
    
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    for i, img in enumerate(images):
        clone = img.copy()
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(img)
        if titles:
            plt.title("{0:.2f}".format(titles[i]))
        plt.xticks([]), plt.yticks([])
    plt.show()



