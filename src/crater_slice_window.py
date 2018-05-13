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
import numpy as np
import cv2 as cv
import glob
from sklearn.model_selection import train_test_split
import argparse
import cv2

(winW, winH) = (40, 40)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

for i in range(4):
	if(i==0):
		image_glob = glob.glob("../crater_dataset/crater_data/images/tile3_24/crater/*.jpg")
	elif(i==1):
		image_glob = glob.glob("../crater_dataset/crater_data/images/tile3_24/non-crater/*.jpg")
	elif(i==2):
		image_glob = glob.glob("../crater_dataset/crater_data/images/tile3_25/crater/*.jpg")
	elif(i==3):
		image_glob = glob.glob("../crater_dataset/crater_data/images/tile3_25/non-crater/*.jpg")
	# loop over the image pyramid
	for images in image_glob:
		image = cv2.imread(images)
		cv2.imshow('Layer', image)
		cv2.waitKey(0)
		for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
		#for resized in pyramid(image, scale=2):
			# loop over the sliding window for each layer of the pyramid
			for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW,winH)):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
# You may need to normalized the window before passing it as input to your classifier
# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW,SUCH AS APPLYING A
# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
# WINDOW
# since we do not have a classifier, we'll just draw the window
				clone = resized.copy()
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)