import tensorflow as tf
import numpy as np

def empty_variable(shape):
	init = tf.constant(0.0, shape = shape)
	return tf.Variable(init)

def weight_variable(shape):
	in_size = shape[-2]
	out_size = shape[-1]
	init = tf.truncated_normal(shape, stddev = 0.1)#tf.sqrt(2.0 / (in_size + out_size)))
	return tf.Variable(init)

def bias_variable(shape):
	init = tf.constant(0.0, shape = shape)
	return tf.Variable(init)

def conv2d(x, W, stride=1):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')

def deconv2d(x, W, shape, stride=1):
	return tf.nn.conv2d_transpose(x, W, shape, strides = [1, stride, stride, 1], 
		padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def config(use_gpu):
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.gpu_options.visible_device_list = str(use_gpu)
	return config
