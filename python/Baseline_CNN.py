# -*- coding: utf-8 -*-

""" 
// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:	8749390300
// email:	wang133@usc.edu

// Compiled on Python 3.5.3 with the tensorflow-gpu 1.1 and tflearn
// solution for Problem 1 (b) baseline CNN
// Directly input "python path-to-this-file" to run the program after inputing the location of the tflearn
"""
from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.initializations
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# K-means initializing (in this question, I revise the source file in the tflearn.initializations "initializations.py")

initiala = tflearn.initializations.KM1()
initialb = tflearn.initializations.KM2()
initialc = tflearn.initializations.KM3()

initials = tflearn.initializations.uniform(shape=None, minval=0, maxval=1, dtype=tf.float32, seed=None)

# Convolutional network building  
                                          
network1 = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network2 = conv_2d(network1, 6, 5, strides=1, padding='valid', weights_init = initiala, activation='relu')# weights_init = initial,
network3 = max_pool_2d(network2, 2)
network4 = conv_2d(network3, 16, 5, strides=1, padding='valid', weights_init = initialb, activation='relu')#, weights_init = initial1
network5 = max_pool_2d(network4, 2)
network6 = fully_connected(network5, 120, weights_init = initialc, activation='relu')#weights_init = initial2, 
#network6 = dropout(network6, 1)
network7 = fully_connected(network6, 84, weights_init = 'xavier', activation='relu')
#network7 = dropout(network7, 1)
network = fully_connected(network7, 10, bias_init = 'zeros', activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/')
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')
model.save('D:/EE569_Assignment/4/cifar10my_model.tflearn')