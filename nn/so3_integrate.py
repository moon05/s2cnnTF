import tensorflow as tf
import lie_learn.spaces.S3 as S3

def so3_integrate(x):

	assert tf.size(x)(-1) == tf.size(x)(-2)
	assert tf.size(-2) == tf.size(x)(-3)

	b = tf.size(-1) // 2

	#assigning "w" here difrectly instead of having a separate function with
	#gpu usage
	w = S3.quadrature_weights(b)
	w = tf.cast(w, tf.float32)

	if isinstance(x, tf.variable):
		w = tf.variable(w)


	x = tf.reduce_sum(x, axis=-1).squeeze(-1)
	x = tf.reduce_sum(x, axis=-1).squeeze(-1)

	sz = tf.size(x)
	x = tf.reshape(x, -1, 2*b)
	w = tf.reshape(w, 2*b, 1)
	x = tf.matmul(x, w).squeeze(-1)
	x = x.reshape(x, sz[:-1])

	return x