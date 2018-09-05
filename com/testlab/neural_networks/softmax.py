import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_sum = np.sum(np.exp(L))
    L_ret = np.divide(L, exp_sum)
    return L_ret