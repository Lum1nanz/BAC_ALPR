import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense

# this module loads the module for train_base_model.py and extract.py

def get_model_1(input_shape):
    model = tf.keras.Sequential(name="LP_extraction")

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))


    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1,activation='sigmoid'))

    return model