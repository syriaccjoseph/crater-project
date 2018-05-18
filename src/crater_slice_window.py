"""
from skimage.transform import pyramid_gaussian
from skimage import io; io.use_plugin('matplotlib')
import argparse
import cv2
image = cv2.imread('C:/CS697/ML/CraterProjectII/crater_project_2/crater_dataset/crater_data/images/normalized_images/crater/TE_tile3_24_025_norm.jpg')
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
# if the image is too small, break from the loop
if resized.shape[0] < 24 or resized.shape[1] < 24:
break
cv2.imshow("Layer {}".format(i + 1), resized)
cv2.waitKey(0)
# show the resized image
#cv2.imshow('image', image)
cv2.imshow("Layer {}".format(i + 1), resized)
cv2.waitKey(0)
"""
import timeit
import random
from skimage.transform import pyramid_gaussian
import skimage.draw
import numpy as np
#import cv2 as cv
import glob
from sklearn.model_selection import train_test_split
import argparse
import cv2

(winW, winH) = (200, 200)

def sliding_window(image, stepSize, windowSize):
# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
		# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# loop over the image pyramid
#for images in image_glob:
image = cv2.imread("../crater_dataset/crater_data/images/tile3_24.pgm", -1)
image2 = cv2.imread("../crater_dataset/crater_data/images/tile3_25.pgm", -1)

cv2.imshow('Layer', image)
cv2.waitKey(0)
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
	#for resized in pyramid(image, scale=2):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW,winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		y_len, x_len = resized.shape
		#cropped_image=resized[int((y*y_len)/winH):int(((y+1)*y_len)/winH),int((x*x_len)/winW):int(((x+1)*x_len)/winW)]
		t_img = resized[y:y+winW,x:x+winH]
		cv2.imshow("cropped Window",t_img)
		
				# SGD.test_mb_accuracy()
				# cv2.circle(image, (100, 100), 100, (0,0,255), 3)
				# cv2.circle(image, (100, 100), 100, (255,0,0), 3)

				# cv2.imwrite('detected_tile3_24.jpg', image)

				# cv2.imwrite('detected_tile3_25.jpg', image)



	# You may need to normalized the window before passing it as input to your classifier
	# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW,SUCH AS APPLYING A
	# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
	# WINDOW
	# since we do not have a classifier, we'll just draw the window



		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)


cv2.imshow('Layer', image2)
cv2.waitKey(0)
for (i, resized) in enumerate(pyramid_gaussian(image2, downscale=2)):
	#for resized in pyramid(image, scale=2):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW,winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		y_len, x_len = resized.shape
		#cropped_image=resized[int((y*y_len)/winH):int(((y+1)*y_len)/winH),int((x*x_len)/winW):int(((x+1)*x_len)/winW)]
		t_img = resized[y:y+winW,x:x+winH]
		cv2.imshow("cropped Window",t_img)
				# SGD.test_mb_accuracy()
				# cv2.circle(image, (100, 100), 100, (0,0,255), 3)
				# cv2.circle(image, (100, 100), 100, (255,0,0), 3)

				# cv2.imwrite('detected_tile3_24.jpg', image)

				# cv2.imwrite('detected_tile3_25.jpg', image)



	# You may need to normalized the window before passing it as input to your classifier
	# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW,SUCH AS APPLYING A
	# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
	# WINDOW
	# since we do not have a classifier, we'll just draw the window



		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)