#import pre-processing.py
import crater_loader
import crater_network
import matplotlib.pyplot as plt




# import cPickle
# import gzip
# import random
# import numpy as np
# import cv2 as cv
# import glob
# import re

det_list = [] 
fal_list = [] 
qty_list = [] 
tp_list = [] 
fp_list = [] 
fn_list = [] 

for x in range(0, 10):

	print "Loading train and test data"

	training_data, test_data = crater_loader.load_data_wrapper()

	print "Training Network"

	#network with 40000 pixel input for 200x200 picture, 10 hidden neurons, 4 hidden neurons, 1 output
	craternn = crater_network.Network([40000, 10, 4, 1])

	print "Running SGD"

	#stochastic gradient descent 




	tp, fp, fn, det2, fal2, qty2 = craternn.SGD(training_data, 60, 5, 0.2, test_data=test_data)
	
	print "TP, FP and FN VALUES"
	print tp
	print fp
	print fn
	print "DETECTION, FALSE RATE, QUALITY, TP, FP and FN VALUES LISTS"
	det_list.append(det2[0][0])
	fal_list.append(fal2[0][0])
	qty_list.append(qty2[0][0])
	tp_list.append(tp[0][0])
	fp_list.append(fp[0][0])
	fn_list.append(fn[0][0])
	print tp_list
	print fp_list
	print fn_list
	print det_list
	print fal_list
	print qty_list

	avg_tp =  sum(tp_list)/len(tp_list)
	avg_fp = sum(fp_list)/len(fp_list)
	avg_fn = sum(fn_list)/len(fn_list)
	avg_det = sum(det_list)/len(det_list)
	avg_fal = sum(fal_list)/len(fal_list)
	avg_qty = sum(qty_list)/len(qty_list)

	print "AVERAGE VALUES"
	print avg_tp
	print avg_fp
	print avg_fn
	print avg_det
	print avg_fal
	print avg_qty


plt.figure(1)
plt.plot(range(10), det_list, 'r--', range(10), fal_list, 'bs', range(10), qty_list, 'g^')
plt.figure(2)
plt.plot(range(10), tp_list, 'b--', range(10), fp_list, 'ys', range(10), fn_list, 'r^')

plt.show()

