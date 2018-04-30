### Libraries
# Standard library

import timeit
import re

# Third-party libraries
import numpy as np

import cv2 as cv
import glob
from sklearn.model_selection import train_test_split

start_time = end_time1 = end_time2 = timeit.default_timer()

def load_data():

    # Getting data from normalize_images folder for crater images

    crater_glob = glob.glob("../crater_dataset/crater_data/images/normalized_images/crater/*.jpg")

    # rgb to grayscale conversion cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY),
    # cv.imread(...) gets data as numpy array
    # np.reshape reshapes it into (40000, 1) one tuple np.array, just like mnist loads data into network
    # the [... for image in  non_crater_glog] is list comprehension in python

    crater_array = [np.reshape(cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) / 255.0, (40000, 1)) for image in crater_glob]

    crater_img = ['TE'.split() + re.split('\.', (re.split('TE', image)[1])) for image in crater_glob]
    # re.split('\.', (re.split('TE', image)[1]))



    # ones corresponding to each entry in crater_array
    crater_ones =  [np.ones((1,1))] * len(crater_array)

    # Getting data from normalize_images folder for non-crater images
    non_crater_glob = glob.glob("../crater_dataset/crater_data/images/normalized_images/non-crater/*.jpg")

    # rgb to grayscale conversion cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY),
    # cv.imread(...) gets data as numpy array
    # np.reshape reshapes it into (40000, 1) one tuple np.array, just like mnist loads data into network
    # the [... for image in  non_crater_glog] is list comprehension in python

    non_crater_array = [np.reshape(cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) / 255.0, (40000, 1)) for image in non_crater_glob]


    non_crater_img = ['FE'.split() + re.split('\.', (re.split('FE', image)[1])) for image in non_crater_glob]
    # zeros corresponding to each entry in non_crater_array
    non_crater_zeros =  [np.zeros((1, 1))] * len(non_crater_array)


    #zipping both crater and non crater stuff
    whole_data = zip(crater_array + non_crater_array, crater_ones + non_crater_zeros, crater_img + non_crater_img)

    train_data, test_data = train_test_split(whole_data, test_size=0.30, random_state=42)
    return train_data, test_data




