
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import object_detector.file_io as file_io
import object_detector.detector as detector
import object_detector.factory as factory
import argparse as ap

DEFAULT_HNM_OPTION = True
DEFAULT_CONFIG_FILE = "conf/svhn.json"

def show(image):
    import cv2
    image = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
    image = image.astype('uint8')
    cv2.imshow("img", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-i', "--include_hnm", help="Include Hard Negative Mined Set", default=DEFAULT_HNM_OPTION, type=bool)
    args = vars(parser.parse_args())
    
    conf = file_io.FileJson().read(args["config"])
    
    #1. Load Features and Labels
    getter = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], 
                                              conf["descriptor"]["parameters"], 
                                              conf["detector"]["window_dim"], 
                                              conf["extractor"]["output_file"])
    getter.summary()
    features, labels = getter.get_dataset(include_hard_negative=args["include_hnm"])    


    import numpy as np
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05, fill_mode='nearest')

    features_pos_aug = []
    labels_pos_aug = []
    
    label_index = range(1, 11)
    for label in label_index:
        
        print label
        
        features_pos = features[labels.reshape(-1,) == label]
        labels_pos = labels[labels.reshape(-1,) == label]

        n_data = features_pos.shape[0]
        
        i = 0
        for batch in datagen.flow(features_pos, batch_size=n_data):
            i += 1
            features_pos_aug.append(batch)
            labels_pos_aug.append(np.zeros((n_data, 1)) + label)
      
            if i >= 10:
                break  # otherwise the generator would loop indefinitely

    features_pos_aug = np.concatenate(features_pos_aug, axis=0)
    labels_pos_aug = np.concatenate(labels_pos_aug, axis=0) + 100
    
    print features_pos_aug.shape
    print labels_pos_aug.shape
    
    getter.add_data(features_pos_aug, 100)
    getter.summary()

    # 3. Save dataset
    getter.save(data_file=conf["extractor"]["output_file"])
       

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    