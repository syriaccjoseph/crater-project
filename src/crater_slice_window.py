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