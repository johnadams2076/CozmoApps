from com.testlab.sobel.applysobel import abs_sobel_thresh
from com.testlab.sobel.magnitudeofgradient import mag_thresh
from com.testlab.sobel.dir_threshold import dir_threshold

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

#Combine thershold
def combine_threshold():

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20,100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20,100))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) & (grady == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


# Run the function
combined = combine_threshold()
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title("Combined Threshold", fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
