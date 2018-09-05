# Solution is available in the "solution.ipynb"
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.subtract(tf.cast(tf.divide(x, y), tf.int32),  1)

# TODO: Print z from a session as the variable output
output = tf.Session().run(z, feed_dict={x: 10, y: 2})
print(output)