# USAGE
# python extract_features.py --conf conf/cars.json

# import the necessary packages
from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
from pyimagesearch.object_detection import helpers
from pyimagesearch.descriptors import HOG
from pyimagesearch.utils import dataset
from pyimagesearch.utils import Conf
from imutils import paths
from scipy import io
import numpy as np
import progressbar
import argparse
import random
import cv2

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the HOG descriptor along with the list of data and labels
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
data = []
labels = []

# grab the set of ground-truth images and select a percentage of them for training
trnPaths = list(paths.list_images(conf["image_dataset"]))
trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))
print("[INFO] describing training ROIs...")

# setup the progress bar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()

# loop over the training paths
for (i, trnPath) in enumerate(trnPaths):
    # load the image, convert it to grayscale, and extract the image ID from the path
    image = cv2.imread(trnPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageID = trnPath[trnPath.rfind("_") + 1:].replace(".jpg", "")

    # load the annotation file associated with the image and extract the bounding box
    p = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
    bb = io.loadmat(p)["box_coord"][0]
    roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))

    # define the list of ROIs that will be described, based on whether or not the
    # horizontal flip of the image should be used
    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

    # loop over the ROIs
    for roi in rois:
        # extract features from the ROI and update the list of features and labels
        features = hog.describe(roi)
        data.append(features)
        labels.append(1)

    # update the progress bar
    pbar.update(i)

# grab the distraction image paths and reset the progress bar
pbar.finish()
dstPaths = list(paths.list_images(conf["image_distractions"]))
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()
print("[INFO] describing distraction ROIs...")

# loop over the desired number of distraction images
for i in np.arange(0, conf["num_distraction_images"]):
    # randomly select a distraction images, load it, convert it to grayscale, and
    # then extract random pathces from the image
    image = cv2.imread(random.choice(dstPaths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(image, tuple(conf["window_dim"]),
        max_patches=conf["num_distractions_per_image"])

    # loop over the patches
    for patch in patches:
        # extract features from the patch, then update teh data and label list
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

    # update the progress bar
    pbar.update(i)

# dump the dataset to file
pbar.finish()
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")