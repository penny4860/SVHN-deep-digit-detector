from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras import layers, activations
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, nb_epoch=5, nb_classes=2, do_augment=False, save_file='models/detector_model.hdf5'):
    """ vgg-like deep convolutional network """
    
    np.random.seed(1337)  # for reproducibility
      
    # input image dimensions
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    
    # size of pooling area for max pooling
    pool_size = (2, 2)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Convolution2D(filters=nb_filters, kernel_size=3, padding='valid',
                            input_shape=input_shape, activation='relu'))
    model.add(Convolution2D(filters=nb_filters, kernel_size=3, 
                            input_shape=input_shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
     
    model.add(Convolution2D(filters=nb_filters*2, kernel_size=3,
                            activation='relu'))
    model.add(Convolution2D(filters=nb_filters*2, kernel_size=3, 
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation(activations.relu))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation(activations.softmax))
        
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    if do_augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2)
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=len(X_train), epochs=nb_epoch,
                            validation_data=(X_test, Y_test))
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)