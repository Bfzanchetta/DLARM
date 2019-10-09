import tensorflow as tf
import pandas as pd

aux = tf.Placeholder(tf.float32)
a = tf.variable(3.0, type=tf.float32)

sess = tf.Session()
print(sess.run(a))