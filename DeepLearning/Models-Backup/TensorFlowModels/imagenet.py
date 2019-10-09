###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Adaptation:   Breno Zanchetta
# Description:  Functions for loading the imagenet image paths and labels into memory.
###############################################################################
import tensorflow as tf
import random
import os

train_dir = "/home/nvidia/Desktop/Desenvolvimento/newset/train"
#validation_dir = "/home/nvidia/Desktop/Desenvolvimento/newset/val"
labels_file = "/home/nvidia/Desktop/Desenvolvimento/ilsvrc12_synset_words.txt"
###############################################################################
# source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py
def _find_image_files(data_dir, labels_file):
    print('Determining list of input files and labels from %s.' % data_dir)
    challenge_synsets = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
    labels = []
    filenames = []
    synsets = []
    # Leave label index 0 empty as a background class.
    label_index = 1
    # Construct the list of JPEG files and labels.
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
        matching_files = tf.gfile.Glob(jpeg_file_path)
        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)
        if not label_index % 100:
                print('Finished finding files in %d of %d classes.' % (label_index, len(challenge_synsets)))
        label_index += 1
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(challenge_synsets), data_dir))
    return filenames, synsets, labels

class imagenet_data:
    train_filenames, train_synsets, train_labels = _find_image_files(train_dir, labels_file)
    #val_filenames, val_synsets, val_labels = _find_image_files(validation_dir, labels_file)

def check_if_downloaded():
    if os.path.exists(train_dir):
        print("Train directory seems to exist")
    else:
        raise Exception("Train directory doesn't seem to exist.")

    #if os.path.exists(validation_dir):
        #print("Validation directory seems to exist")
    #else:
        #raise Exception("Validation directory doesn't seem to exist.")

def load_class_names():
    return data.humans

def load_training_data():
    return data.train_filenames, data.train_labels

#def load_test_data():
    #return data.val_filenames, data.val_labels

data = imagenet_data()
