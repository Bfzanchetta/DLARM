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
from imgnet import imagenet_data
warnings.filterwarnings("ignore")

epochs=10
channels=3
num_threads=4
num_classes=1000

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

d = imagenet_data(batch_size=4, sess=sess)
image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

#Definicao do modelo
#modelo to-do
#fim modelo

loss_fc = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(target_batch_tensor, tf.float32), name="cross-entropy")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

since = time.time()

for i in range(epochs):
        epoch_time = time.time()
        print("Epoch ", i)
        for j, (input, targets) in enumerate(sess.run([image_batch_tensor, target_batch_tensor])):
                print("Batch numero ", j)
print("Fim")
