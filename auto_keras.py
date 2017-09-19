import numpy as np
import cv2
import glob
from random import shuffle
import tensorflow as tf
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from quiver_engine import server

height_input = 20 #height of the input image of the NN
width_input = 20  #width  of the input image of the NN
color = False   #True means images will be used as bgr, False as grayscale
folder_name = 'imagesets'
train_ratio = 1 #Percentage of pictures used for train set
if color == True:
	depth = 3
else:
	depth = 1

def load_sets():
	true == false
	x_set = np.load('x_set.npy')
	y_set = np.load('y_set.npy')
	classes = np.load('variable_names.npy')
	print 'classes : {}'.format(classes)
	print ('sets loaded')
	return x_set, y_set

def count_examples(folder_names):
	tot_pictures = 0
	for folder in folder_names:
		pictures = folder+'/*'
		for picture in glob.glob(pictures):
			tot_pictures+=1
	return tot_pictures

def create_variable_names(folder_names, name_size):
	classes = []
	for folder in folder_names:
		classes.append(folder[name_size+1:])
	np.save('variable_names.npy',classes)
	print ('classes saved')
	return classes

def load_image(height, width, picture, color):
	if color == True:
		img = cv2.imread(picture)
		img = cv2.resize(img,(height,width))
	else:
		img = cv2.imread(picture,0)
		img = cv2.resize(img,(height,width))
		img = np.expand_dims(img, axis = -1)
	cv2.imwrite('0.png',img)
	return img

def create_sets(folder_name, height, width, color):
	folder_names = glob.glob(folder_name+'/*')
	name_size = len(folder_name)
	classes = create_variable_names(folder_names, name_size)
	print 'classes : {}'.format(classes)
	tot_pictures = count_examples(folder_names)
	
	y_set = np.zeros((tot_pictures,len(classes)))
	if color == True:
		x_set = np.zeros((tot_pictures, height, width, 3))
	elif color == False:
		x_set = np.zeros((tot_pictures, height, width, 1))
	else:
		print 'color must be True or False'

	class_nb = 0
	tot = 0
	for i in range(len(folder_names)):
		pictures = folder_names[i]+'/*'
		for picture in glob.glob(pictures):
			x_set[tot,:,:,:] = load_image(height, width, picture,color)
			y_set[tot,class_nb] = 1
			tot +=1
		class_nb +=1
	
	np.save('x_set.npy',x_set)
	np.save('y_set.npy',y_set)
	print ('sets saved')	
	return x_set, y_set

def create_train_test(x_set, y_set, train_ratio):
	nb_examples = y_set.shape[0]
	train_nb = int(train_ratio*nb_examples)
	test_nb = nb_examples - train_nb
	x_train_shape = (train_nb,)+x_set.shape[1:]
	x_test_shape  = (test_nb,) +x_set.shape[1:]
	y_train_shape = (train_nb,)+y_set.shape[1:]
	y_test_shape  = (test_nb,) +y_set.shape[1:]

	random_indices = range(nb_examples)
	shuffle(random_indices)

	x_train = np.zeros(x_train_shape)
	x_test = np.zeros(x_test_shape)
	y_train = np.zeros(y_train_shape)
	y_test = np.zeros(y_test_shape)

	for cpt in range(len(random_indices)):
		if cpt < train_nb:
			x_train[cpt,:,:,:] = x_set[random_indices[cpt],:,:,:]
			y_train[cpt,:]     = y_set[random_indices[cpt],:]
		else:
			x_test[cpt-train_nb,:,:,:] = x_set[random_indices[cpt-train_nb],:,:,:]
			y_test[cpt-train_nb,:]     = y_set[random_indices[cpt-train_nb],:]
	return x_train, x_test, y_train, y_test			

try:
	x_set,y_set = load_sets()
except:
	x_set, y_set = create_sets(folder_name, height_input, width_input, color)
	

X_train, test_set, Y_train, y_test = create_train_test(x_set, y_set, train_ratio)
print test_set.shape, y_test.shape


model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(20, 20,1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(Y_train.shape[1]))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam)

model.fit(X_train, Y_train, validation_split = 0.2, batch_size=50, nb_epoch=2, show_accuracy=True)
server.launch(model)
