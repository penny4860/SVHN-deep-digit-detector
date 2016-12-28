
import object_detector.file_io as file_io
import object_detector.detector as detector
import object_detector.factory as factory
import argparse as ap

DEFAULT_CONFIG_FILE = "conf/svhn.json"

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    args = vars(parser.parse_args())
    
    conf = file_io.FileJson().read(args["config"])
    annotation_filename=conf["dataset"]["annotation_file"]

    import cv2
    import os
    
    image_path = os.path.split(annotation_filename)[0]
    annotations = file_io.FileJson().read(annotation_filename)
    negative_path = os.path.join(image_path, "negative_images")
    if os.path.exists(negative_path) == False:
        os.mkdir(negative_path)
    
    for annotation in annotations:
        image = cv2.imread(os.path.join(image_path, annotation["filename"]))
        for box in annotation["boxes"]:
            x1 = int(box["left"])
            y1 = int(box["top"])
            w = int(box["width"])
            h = int(box["height"])
            bb = (y1, y1+h, x1, x1+w)
            image[y1:y1+h, x1:x1+w, :] = 0
    
        cv2.imwrite(os.path.join(negative_path, annotation["filename"]), image)
    
    print "done"
