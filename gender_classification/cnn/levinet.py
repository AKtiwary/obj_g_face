from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeviNet:
    @staticmethod
    def build(width, height, channel, classes):
        input_shape = (height, width, channel)
        channel_dim = -1

        model = Sequential()

        # Block 1:
        # CONV => RELU => POOL
        model.add(Conv2D(96, (7, 7), strides=(4, 4), input_shape=input_shape) )
        model.add(Activation('relu') )
        
        model.add(BatchNormalization(axis=channel_dim) )

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2) ) )

        model.add(Dropout(0.25) )

        # Block 2:
        # CONV => RELU => POOL
        model.add(Conv2D(256, (5, 5), padding='valid') )
        model.add(Activation('relu') )

        model.add(BatchNormalization(axis=channel_dim) )

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2) ) )

        model.add(Dropout(0.25) )

        # Block 3:
        # CONV => RELU => POOL
        model.add(Conv2D(384, (3, 3), padding='valid') )
        model.add(Activation('relu') )

        model.add(BatchNormalization(axis=channel_dim) )

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2) ) )

        model.add(Dropout(0.25) )

        # Block 4:
        # FC => RELU
        model.add(Flatten() )
        model.add(Dense(512) )
        model.add(Activation('relu') )
        model.add(BatchNormalization(axis=channel_dim) )
        model.add(Dropout(0.5) )

        # Block 5:
        # FC => RELU
        model.add(Dense(512) )
        model.add(Activation('relu') )
        model.add(BatchNormalization(axis=channel_dim) )
        model.add(Dropout(0.5) )

        # Final:
        model.add(Dense(classes) )
        model.add(Activation('softmax') )

        return model