
# coding: utf-8

# In[ ]:


# Solution is available in the other "solution.py"
import tensorflow as tf
from test import *
tf.set_random_seed(123456)


hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]


# In[ ]:


# set random seed
tf.set_random_seed(123456)


# In[ ]:


# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]


# In[ ]:


# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])


# In[ ]:


# TODO: Create Model with Dropout
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=0.5)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])


# In[ ]:


# TODO: save and print session results as variable named "output"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(logits)
    print(output)

# In[ ]:


### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(output)
except Exception as err:
    print(str(err))

