import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -np.sum((Y * np.log(P) + np.subtract(1,Y) * np.log(np.subtract(1, P))))
Y = [1,1,0,1,0]
P = [0.6, 0.7,0.7,0.96, 0.35]
print(cross_entropy(Y, P))