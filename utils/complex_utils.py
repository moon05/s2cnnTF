import tensorflow as tf
import numpy as np


#function would take in real-valued array
#return as complex array with real part as the inital array
#imaginary part as zeros

def as_complex(x):
	imaginary = tf.zeros([1, tf.size(x)], tf.float32)[0]
	complex_array = tf.complex(x, imaginary)
	return complex_array

def fftshift(x, axis):
	n = tf.size(x)
	return 0


sess = tf.Session()


a = tf.constant([1.0, 2.0, 3.0])

(sess.run(as_complex(a)))