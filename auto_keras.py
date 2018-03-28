import numpy as np
import cv2
import glob
from random import shuffle
import tensorflow as tf
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
#Convolution2D, MaxPooling2D,
from keras.optimizers import Adam
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

height_input = 100 #height of the input image of the NN
width_input = 100  #width  of the input image of the NN
color = True   #True means images will be used as bgr, False as grayscale
folder_name = 'data_set'
train_ratio = 1 #Percentage of pictures used for train set
val_split = 0.2
if color == True:
	depth = 3
else:
	depth = 1

def create_variable_names(folder_names, name_size):
	classes = []
	for folder in folder_names:
		classes.append(folder[name_size+1:])
	np.save('variable_names.npy',classes)
	print ('classes saved')
	return classes

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')


model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
#model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(height_input, width_input,depth)))
model.add(Conv2D(32, (3,3),input_shape=(height_input, width_input,depth), strides=(1, 1), padding='valid', data_format="channels_last", dilation_rate=(1, 1), activation='relu'))
#, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
#model.add(Activation('relu'))
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Convolution2D(64, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(Y_train.shape[1]))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)

#server.launch(model)
