###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Adaptation:   Breno Zanchetta
# Description:  Functions for loading the imagenet image paths and labels into memory.
###############################################################################
import tensorflow as tf
import random
import os

train_dir         = "data/imagenet/train/"
validation_dir    = "data/imagenet/validation/"
labels_file       = "data/imagenet/imagenet_lsvrc_2015_synsets.txt"
metadata_file     = "data/imagenet/imagenet_metadata.txt"
bounding_box_file = "data/imagenet/imagenet_2012_bounding_boxes.csv"

###############################################################################
# Some TensorFlow Inception functions (ported to python3)
# source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py

def _find_image_files(data_dir, labels_file):
  
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in
                       tf.gfile.FastGFile(labels_file, 'r').readlines()]

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
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
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

def _find_human_readable_labels(synsets, synset_to_human):
  humans = []
  for s in synsets:
    assert s in synset_to_human, ('Failed to find: %s' % s)
    humans.append(synset_to_human[s])
  return humans

def _find_image_bounding_boxes(filenames, image_to_bboxes):
  num_image_bbox = 0
  bboxes = []
  for f in filenames:
    basename = os.path.basename(f)
    if basename in image_to_bboxes:
      bboxes.append(image_to_bboxes[basename])
      num_image_bbox += 1
    else:
      bboxes.append([])
  print('Found %d images with bboxes out of %d images' % (
      num_image_bbox, len(filenames)))
  return bboxes

def _build_synset_lookup(imagenet_metadata_file):
  lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
  synset_to_human = {}
  for l in lines:
    if l:
      parts = l.strip().split('\t')
      assert len(parts) == 2
      synset = parts[0]
      human = parts[1]
      synset_to_human[synset] = human
  return synset_to_human

def _build_bounding_box_lookup(bounding_box_file):
  lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes
  

class imagenet_data:
  synset_to_human = _build_synset_lookup(metadata_file)
  image_to_bboxes = _build_bounding_box_lookup(bounding_box_file)

  val_filenames, val_synsets, val_labels = _find_image_files(validation_dir, labels_file)
  train_filenames, train_synsets, train_labels = _find_image_files(train_dir, labels_file)
  humans = _find_human_readable_labels(val_synsets, synset_to_human)

def check_if_downloaded():
  if os.path.exists(train_dir):
    print("Train directory seems to exist")
  else:
    raise Exception("Train directory doesn't seem to exist.")

  if os.path.exists(validation_dir):
    print("Validation directory seems to exist")
  else:
    raise Exception("Validation directory doesn't seem to exist.")


def load_class_names():
  return data.humans

def load_training_data():
  return data.train_filenames, data.train_labels

def load_test_data():
  return data.val_filenames, data.val_labels

data = imagenet_data()
