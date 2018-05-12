### Libraries
# Standard library

import timeit
import random

# Third-party libraries
import numpy as np

import cv2 as cv
import glob
from sklearn.model_selection import train_test_split

start_time = end_time1 = end_time2 = timeit.default_timer()

def load_data_wrapper():

    # Getting data from normalize_images folder for crater images

    crater_glob = glob.glob("../crater_dataset/crater_data/images/normalized_images/crater/*.jpg")
    # rgb to grayscale conversion cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY), cv.imread(...) gets data as numpy array
    # np.reshape reshapes it into (40000, 1) one tuple np.array, just like mnist loads data into network
    # the [... for image in  non_crater_glog] is list comprehension in python

    crater_array = [np.reshape(cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) / 255.0, (40000, 1)) for image in crater_glob]

    # ones corresponding to each entry in crater_array
    crater_ones =  [np.ones((1,1))] * len(crater_array)

    # Getting data from normalize_images folder for non-crater images
    non_crater_glob = glob.glob("../crater_dataset/crater_data/images/normalized_images/non-crater/*.jpg")
    # rgb to grayscale conversion cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY), cv.imread(...) gets data as numpy array
    # np.reshape reshapes it into (40000, 1) one tuple np.array, just like mnist loads data into network
    # the [... for image in  non_crater_glog] is list comprehension in python

    non_crater_array = [np.reshape(cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) / 255.0, (40000, 1)) for image in non_crater_glob]
    # zeros corresponding to each entry in non_crater_array
    non_crater_zeros =  [np.zeros((1, 1))] * len(non_crater_array)


    #zipping both crater and non crater stuff
    whole_data = zip(crater_array + non_crater_array, crater_ones + non_crater_zeros)
    train_data, test_data = train_test_split(whole_data, test_size=0.30, random_state=42)
    return train_data, test_data



def load_crater_data_phaseII_wrapper():

    # Getting data from normalize_images folder for crater images

    crater_glob = glob.glob("../crater_dataset/crater_data/images/normalized_images/crater/*.jpg")
    # rgb to grayscale conversion cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY), cv.imread(...) gets data as numpy array
    # np.reshape reshapes it into (40000, 1) one tuple np.array, just like mnist loads data into network
    # the [... for image in  non_crater_glog] is list comprehension in python

    crater_array = [np.reshape(cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) / 255.0, (1, 40000)) for image in crater_glob]
    #crater_array = [cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) for image in crater_glob]
    #print len(crater_array[0][0]
    # ones corresponding to each entry in crater_array
    crater_ones =  [np.ones((1,1))] * len(crater_array)
    #crater_ones =  np.ones((len(crater_array),1)) 
    #print crater_ones
    # Getting data from normalize_images folder for non-crater images
    non_crater_glob = glob.glob("../crater_dataset/crater_data/images/normalized_images/non-crater/*.jpg")
    # rgb to grayscale conversion cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY), cv.imread(...) gets data as numpy array
    # np.reshape reshapes it into (40000, 1) one tuple np.array, just like mnist loads data into network
    # the [... for image in  non_crater_glog] is list comprehension in python

    non_crater_array = [np.reshape(cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) / 255.0, (1, 40000)) for image in non_crater_glob]
    #non_crater_array = [cv.cvtColor(cv.imread(image), cv.COLOR_RGB2GRAY) for image in non_crater_glob]
    # zeros corresponding to each entry in non_crater_array
    non_crater_zeros =  [np.zeros((1, 1))] * len(non_crater_array)
    #non_crater_zeros =  np.zeros((len(non_crater_array),1))
    
    #zipping both crater and non crater stuff
    whole_data = zip(crater_array + non_crater_array, crater_ones + non_crater_zeros)
    #print whole_data[0][0][0]
    #for i in range(len(whole_data)):
    #   crater_array_labels.append(whole_data[i][1][0][0])
    #print crater_array_list
    train_data, validation_data = train_test_split(whole_data, test_size=0.15, random_state=42)
    train_data, test_data = train_test_split(train_data, test_size=0.1275, random_state=42)
	
	
    crater_array_label = []
    crater_array_data = []	
	
    for i in range(len(train_data)):
        crater_array_label.append(train_data[i][1][0][0])
    for i in range(len(train_data)):
        crater_array_data.append(train_data[i][0][0])
    training_data = zip([crater_array_data],[crater_array_label])
    #print training_data
    crater_array_label = []
    crater_array_data = []		

    for i in range(len(validation_data)):
        crater_array_label.append(validation_data[i][1][0][0])
    for i in range(len(validation_data)):
        crater_array_data.append(validation_data[i][0][0])
    validating_data = zip([crater_array_data],[crater_array_label])
	
    crater_array_label = []
    crater_array_data = []	
	
    for i in range(len(test_data)):
        crater_array_label.append(test_data[i][1][0][0])
    for i in range(len(test_data)):
        crater_array_data.append(test_data[i][0][0])
    testing_data = zip([crater_array_data],[crater_array_label])
    #print testing_data
    #return train_data, validation_data, test_data
    #print "TRAIN DATA"
    #print train_data
    #print "VALIDATION DATA"
    #print validation_data
    #print "TEST DATA"
    #print test_data
    #print len(train_data)
    #print len(validation_data)
    #print len(test_data)
    return training_data,validating_data,testing_data	

#load_crater_data_phaseII_wrapper()
	
