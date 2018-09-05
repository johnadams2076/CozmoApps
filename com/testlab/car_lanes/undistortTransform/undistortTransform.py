import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8  # the number of inside corners in x
ny = 6  # the number of inside corners in y

plt.interactive(False)
# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found:
    # a) draw corners
    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    # Note: you could pick any four of the detected corners
    # as long as those four corners define a rectangle
    # One especially smart way to do this would be to use four well-chosen
    # corners that were automatically detected during the undistortion steps
    # We recommend using the automatic detection of corners in your code
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    # delete the next two lines
    # M = None
    # warped = np.copy(img)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        print(corners)
        x1, y1 = corners[0][0]
        x2, y2 = corners[1][0]
        x3, y3 = corners[nx][0]
        x4, y4 = corners[nx + 1][0]
        src = np.float32([[corners[0][0]], [corners[1][0]], [corners[nx][0]], [corners[nx + 1][0]]])
        dst = np.float32([[x1 - 300, y1], [x2 - 300, y1], [x1 - 300, y3], [x2 - 300, y3]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, M


top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show(block=True)
plt.interactive(False)