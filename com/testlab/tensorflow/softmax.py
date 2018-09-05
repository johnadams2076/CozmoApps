# Solution is available in the other "solution.py" tab
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=0))

# logits is a two-dimensional array
logits2 = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
logits1 = [3.0, 1.0, 0.2]
print(softmax(logits2))