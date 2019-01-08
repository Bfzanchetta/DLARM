from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from skimage import io, transform

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import time
import argparse
import random
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")

epochs=10
channels=3
num_threads=4
num_classes=1000


#https://github.com/ischlag/tensorflow-input-pipelines/blob/master/datasets/imagenet.py
#https://www.tensorflow.org/guide/datasets
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

#to be continued
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
