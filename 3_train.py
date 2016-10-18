#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import object_detector.factory as factory
import argparse as ap

DEFAULT_HNM_OPTION = True
DEFAULT_CONFIG_FILE = "conf/svhn.json"

def preprocess(features, labels, nb_classes=2):
    from sklearn.model_selection import train_test_split
    from keras.utils import np_utils
    X = features.reshape(-1, 32, 16, 1)
    y = labels.astype(int)
    y[y > 0] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')    
    mean_value = X_train.mean()
    X_train -= mean_value
    X_test -= mean_value
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test, mean_value
    

def imshow(X):
    img = X.reshape(32, 16)
    import cv2
    img = cv2.resize(img, (100, 200), interpolation=cv2.INTER_AREA)
    img = img.astype('uint8')
    cv2.imshow("", img)
    cv2.waitKey(0)


def train_detector(X_train, X_test, Y_train, Y_test, save_file='models/detector_model.hdf5'):
    import numpy as np
    np.random.seed(1337)  # for reproducibility
      
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras import backend as K
      
    batch_size = 128
    nb_epoch = 2
      
    # input image dimensions
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    input_shape = (img_rows, img_cols, 1)


    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (16, 8, 32)
     
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (8, 4, 64) = (2048)
        
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
        
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
        
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)  
    

def load_model(filename):
    from keras.models import load_model
    model = load_model(filename)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model


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

    X_train, X_test, Y_train, Y_test, mean_value = preprocess(features, labels, 2)
    train_detector(X_train, X_test, Y_train, Y_test, "temp.hdf5")
     



    
