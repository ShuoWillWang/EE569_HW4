# -*- coding: utf-8 -*-

""" 
// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:		8749390300
// email:	wang133@usc.edu

// Compiled on Python 3.5.3 with the tensorflow-gpu 1.1 and tflearn
// solution for Problem 2 (a) Our approach to increase the accuracy
// Directly input "python path-to-this-file" to run the program after inputing the location of the tflearn
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d , global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import batch_normalization

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 96, 3, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)#batch normalization
network = conv_2d(network, 96, 3, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 96, 3, activation='relu',weights_init ='xavier',strides = 2)
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 192, 3, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 192, 3, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 192, 3, activation='relu',weights_init ='xavier',strides = 2)
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 192, 3, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 192, 1, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 96, 1, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = conv_2d(network, 10, 1, activation='relu',weights_init ='xavier')
network = tflearn.layers.normalization.batch_normalization (network)
network = tflearn.global_avg_pool(network)#global averaging pool
network = fully_connected(network, 10, activation='softmax',weights_init ='xavier')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/home/will/cifar10_re/tflearn_logs/Pro2_SecondL3')
model.fit(X, Y, n_epoch = 300, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=256, run_id='cifar10_cnn')
