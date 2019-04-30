#!/usr/local/bin/python
# coding: latin-1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys


import numpy as np
import tensorflow as tf
import csv
import Image
import numpy as np
from scipy.misc import imread
from os import listdir
from os.path import isfile, join
from skimage import io
import glob, os         

tf.logging.set_verbosity(tf.logging.INFO)

features = np.array[784,1]
labels = np.array[784,1]

def cnn_model_fn(features, labes, mode):
	#This part is for the actual transformations#
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	#Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	#Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

	#Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	#Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

	# Flatten tensor into a batch of vectors
  	# Input Tensor Shape: [batch_size, 7, 7, 64]
  	# Output Tensor Shape: [batch_size, 7 * 7 * 64]
  	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  	# Dense Layer
  	# Densely connected layer with 1024 neurons
  	# Input Tensor Shape: [batch_size, 7 * 7 * 64]
  	# Output Tensor Shape: [batch_size, 1024]
  	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  	# Add dropout operation; 0.7 probability that element will be kept
  	dropout = tf.layers.dropout(
      inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  	# Logits layer
  	# Input Tensor Shape: [batch_size, 1024]
  	# Output Tensor Shape: [batch_size, 26]
  	logits = tf.layers.dense(inputs=dropout, units=26)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	#Aqui vou chamar a função que lê o dataset e cria o dataframe/feature_column
	#Aqui vou passar o feature column para o train_fn
	for i in range(0, 20):
		os.chdir("/home/breno/Desktop/Final/letter"+str(chr(j+97))+"/")
		for file in glob.glob("*.jpg"):
			im = Image.open(file).convert('L')
			im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
			im_arr = im_arr.reshape((im.size[1], im.size[0], 1))
			np.dstack(features,im_arr)
			np.dstack(labels,i)

	assert features.shape[0] == labels.shape[0]

	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	print("Works so far")	

	#Quando o data_set estiver pronto, separar entre training e eval data sets.
	#O images é a referência da primeira coluna do CSV, a label é a segunda coluna.
	train_data = mnist.train.images  # Returns np.array
  	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  	eval_data = mnist.test.images  # Returns np.array
  	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	#Passa a função do modelo como parâmetro ao Estimator e cria um
	#arquivo temporário na pasta tmp
	mnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

	# Set up logging for predictions
  	# Log the values in the "Softmax" tensor with label "probabilities"
  	tensors_to_log = {"probabilities": "softmax_tensor"}
  	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=3,
		shuffle=True)
	#Essa parte eu não entendo se posso tirar.
	mnist_classifier.train(
		input_fn=train_input_fn,
		steps=20000,
		hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)


if __name__ == "__main__":
	tf.app.run()