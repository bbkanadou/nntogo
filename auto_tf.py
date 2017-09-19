import numpy as np
import cv2
import glob
from random import shuffle
import tensorflow as tf
import os
import time

height_input = 20 #height of the input image of the NN
width_input = 20  #width  of the input image of the NN
color = False   #True means images will be used as bgr, False as grayscale
folder_name = 'imagesets'
train_ratio = 0.8 #Percentage of pictures used for train set
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
	

train_set, test_set, y_train, y_test = create_train_test(x_set, y_set, train_ratio)

print y_train[-1]

batch_size = 18
test_size = 50

def change_brightness(img, factor):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s ,v = cv2.split(hsv_img)
    v = (v*factor).astype(np.uint8)
    new_img = cv2.merge((h,s,v))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    return new_img

def init_weights(shape,nem):
    return tf.Variable(tf.random_normal(shape, stddev=0.01),name=nem)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    l1 = tf.nn.relu(conv2d(X, w1))

    l2a = tf.nn.relu(conv2d(l1, w2))
    l2 = max_pool_2x2(l2a)
    l2 = tf.reshape(l2, [-1, w3.get_shape().as_list()[0]])    
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.nn.relu(tf.matmul(l2, w3))
    l3 = tf.nn.dropout(l3, p_keep_hidden)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)

    return pyx

# Launch the graph in a session
first_graph = tf.Graph()
with tf.Session(graph=first_graph) as sess:


    X = tf.placeholder("float", [None, height_input, width_input, depth],"X")
    Y = tf.placeholder("float", [None, 18],"Y")

    w = init_weights([4, 4, 1, 40],"w1")       
    w2 = init_weights([2, 2, 40, 40],"w2")   
    w3 = init_weights([40 * 10* 10, 512],"w3")
    w4 = init_weights([512, 512],"w4") 
    w_o = init_weights([512, 18],"wo")        

    p_keep_conv = tf.placeholder("float",[],"p_keep_input")
    p_keep_hidden = tf.placeholder("float",[],"p_keep_hidden")
    py_x = (model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden))


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
    predict_op = tf.nn.softmax(py_x)


    tf.initialize_all_variables().run()

    os.system("rm -rf /home/benoit/Desktop/templates/lvls/graph")
    tf.train.write_graph(sess.graph_def, "/home/benoit/Desktop/templates/lvls/graph", "graph.pb", False) #proto

    save_path="models/model.ckpt"
    saver = tf.train.Saver()
    #saver.restore(sess, save_path)
    
    for i in range(10000):
	start_time = time.time()
        train_indices = np.arange(len(train_set)) # Get A train Batch
        np.random.shuffle(train_indices)

        for j in range (100):
	    start,end = [batch_size*j,batch_size*j+batch_size]
            sess.run(train_op, feed_dict={X: train_set[train_indices[start:end]], Y: y_train[train_indices[start:end]], p_keep_conv: 0.8, p_keep_hidden: 0.5})
            time.sleep(0.2)

	elapsed_time = time.time() - start_time 
	print(elapsed_time)           

        test_indices = np.arange(len(test_set)) # Get A Test Batch
        np.random.shuffle(train_indices)

	print(i, np.mean(np.argmax(y_test[test_indices], axis=1) ==sess.run(tf.argmax(predict_op,1), feed_dict={X: test_set[test_indices],Y: y_test[test_indices],p_keep_conv: 1.0,p_keep_hidden: 1.0})))


        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        saver.save(sess,save_path)
        print "saved"
        print i     
