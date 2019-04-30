########################################################
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import time
from keras.utils import np_utils
import os
import tensorflow as tf
import shuffle
########################################################
def get_im_cv2(path):
	img = cv2.imread(path)
	resized = cv2.rezise(img, (64,64), cv2.INTER_LINEAR)
	return resized
########################################################
def load_train():
	X_train = []
	Y_train = []
	y_train = []
	start_time = time.time()
	
	print('Read train images')
	folders = ['Type_1', 'Type_2', 'Type_3']
	for fld in folders:
		index = folders.index(fld)
		print('Load folder {} (Index: {})'.format(fld, index))
		path = os.path.join.('.', 'Downloads', 'Intel', 'train', fld, '*.jpg')
		files = glob.glob(path)
		
		for fl in files:
			flbase = os.path.basename(fl)
			img = get_im_cv2(fl)
			X_train.append(img)
			X_train_id.append(flbase)
			y_train.append(index)
			
	print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
	return X_train, y_train, X_train_id
########################################################
##    LOAD the test images
########################################################


			