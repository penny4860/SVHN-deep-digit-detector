#-*- coding: utf-8 -*-
import cv2
import sklearn.feature_extraction.image as skimg

def crop_bb(image, bb, padding=10, dst_size=(32, 32)):
    """Crop patches from an image with desired bounding box.

    Parameters
    ----------
    image : array, shape (n_rows, n_cols, n_channels) or (n_rows, n_cols)
        Input image to crop.
        
    bb : tuple, (y1, y2, x1, x2)
        Desired bounding box.
    
    dst_size : tuple, (h_size, w_size)
        Desired size for returning bounding box.
    
    Returns
    ----------
    patches : array, shape of dst_size

    Examples
    --------
    >>> from skimage import data
    >>> img = data.camera()        # Get Sample Image
    >>> patch = crop_bb(img, (0,10, 10, 20), 2, (6,6))
    >>> patch
    array([[157, 157, 158, 157, 157, 158],
           [158, 158, 158, 158, 158, 156],
           [157, 158, 158, 157, 158, 156],
           [158, 158, 158, 158, 158, 156],
           [158, 156, 155, 155, 157, 155],
           [157, 155, 156, 156, 156, 152]], dtype=uint8)
    
    """
    
    h = image.shape[0]
    w = image.shape[0]

    (y1, y2, x1, x2) = bb
    
    (x1, y1) = (max(x1 - padding, 0), max(y1 - padding, 0))
    (x2, y2) = (min(x2 + padding, w), min(y2 + padding, h))
    
    patch = image[y1:y2, x1:x2]
    patch = cv2.resize(patch, dst_size, interpolation=cv2.INTER_AREA)
 
    return patch

def crop_random(image, dst_size=(32, 32), max_patches=5):
    """Randomly crop patches from an image as desired size.

    Parameters
    ----------
    image : array, shape (n_rows, n_cols, n_channels) or (n_rows, n_cols)
        Input image to crop.
        
    dst_size : tuple, (h_size, w_size)
        Desired size for croppig.
    
    max_patches : int
        Desired number of patches to crop
    
    Returns
    ----------
    patches : array, shape (max_patches, n_rows, n_cols, n_channels) or (max_patches, n_rows, n_cols)

    Examples
    --------
    >>> import numpy as np
    >>> import sklearn.feature_extraction.image as skimg
    >>> one_image = np.arange(100).reshape((10, 10))
    >>> patches = crop_random(one_image, (5,5), 2)
    >>> patches.shape
    (2L, 5L, 5L)
    
    """
    
    patches = skimg.extract_patches_2d(image, 
                                       dst_size,
                                       max_patches=max_patches)
    return patches


def get_file_id(filename):
    """Get file id from filename which has "($var)_($id).($extension)" format.
    ($var) and ($extension) can be allowed anything format.
    
    Parameters
    ----------
    filename : str
        Input filename to extract id
        
    Returns
    ----------
    file_id : str
        ($id) from "($var)_($id).($extension)" format

    Examples
    --------
    >>> filename = "C:\Windows\System32\cmd.exe\image_0122.jpg"
    >>> get_file_id(filename)
    '0122'

    """
    file_id = filename[filename.rfind("_") + 1:filename.rfind(".")]
    return file_id

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    

