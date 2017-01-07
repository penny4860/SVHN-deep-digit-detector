import digit_detector.file_io as file_io
import numpy as np
import os
import cv2
DIR = ""


def train_detector(X_train, X_test, Y_train, Y_test, save_file='models/detector_model.hdf5'):
    import numpy as np
    np.random.seed(1337)  # for reproducibility
      
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras import backend as K
      
    batch_size = 128
    nb_epoch = 10
      
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


if __name__ == "__main__":

    images_train = file_io.FileHDF5().read(os.path.join(DIR, "train.hdf5"), "images")
    labels_train = file_io.FileHDF5().read(os.path.join(DIR, "train.hdf5"), "labels")
    
    images_val = file_io.FileHDF5().read(os.path.join(DIR, "val.hdf5"), "images")
    labels_val = file_io.FileHDF5().read(os.path.join(DIR, "val.hdf5"), "labels")
    
    print images_train.shape, labels_train.shape
    print images_val.shape, labels_val.shape
    
    import digit_detector.preprocess as preproc
    X_train, X_val, Y_train, Y_val, mean_value = preproc.GrayImgTrainPreprocessor().run(images_train, labels_train, images_val, labels_val)
    print mean_value    # 107.524
    
    print X_train.shape, X_val.shape
     
#     train_detector(X_train, X_val, Y_train, Y_val, 'detector_model.hdf5')





