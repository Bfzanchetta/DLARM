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


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import imagenet

imagenet.

#Definicao do modelo
#modelo to-do
#fim modelo

since = time.time()

for i in range(epochs):
        epoch_time = time.time()
        print("Epoch ", i)
        for j, (input, targets) in enumerate(sess.run([image_batch_tensor, target_batch_tensor])):
                print("Batch numero ", j)
