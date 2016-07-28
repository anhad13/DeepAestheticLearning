from cnn_trainer import *
import urllib
import os
import cv2
import numpy
import sys
def trainly(doc_root, no_of_epochs):
	trX, trY= load_training_set(doc_root)
	trX = trX.reshape(-1, 3, 250, 250)
	X = T.ftensor4()
	Y = T.fmatrix()
	w = init_weights((10, 3, 3, 3))
	w2 = init_weights((20, 10, 3, 3))
	w3 = init_weights((2, 20, 3, 3))
	w4 = init_weights((1800, 50))
	w_o = init_weights((50, 2))
	noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4,w_o, 0.2, 0.5)
	l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o,0., 0.)
	y_x = T.argmax(py_x, axis=1)
	cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
	params = [w, w2, w3, w4, w_o]
	updates = RMSprop(cost, params, lr=0.001)
	train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
	predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
	for i in range(int(no_of_epochs)):
		print "Staring Epoch: "+str(i)
		for start, end in zip(range(0, len(trX), 10), range(10, len(trX), 10)):
			print "Processing batch: "+str(start)+" to "+str(end)
			cost = train(trX[start:end], trY[start:end])
	return predict		
def load_training_set(doc_root):
	file=open("AVA.txt","r")
	lines=file.readlines()
	ratings_hash={}
	for line in lines:
		arr=line[:-2].split(' ')
		id1=arr[1]
		if os.path.exists(doc_root+"/"+id1+".jpg")==True:
			average=1*int(arr[2])+2*int(arr[3])+3*int(arr[4])+4*int(arr[5])+5*int(arr[6])+6*int(arr[7])+7*int(arr[8])+8*int(arr[9])+9*int(arr[10])+10*int(arr[11])
			average=average/float(int(arr[2])+int(arr[3])+int(arr[4])+int(arr[5])+int(arr[6])+int(arr[7])+int(arr[8])+int(arr[9])+int(arr[10])+int(arr[11])+int(arr[12]))
			ratings_hash[id1]=average
	X=[]
	Y=[]
	for k in ratings_hash.keys():
		im = cv2.imread(doc_root+'/'+k+'.jpg')
		if im!=None:
			X.append(im.reshape(250*250*3))
			if ratings_hash[k]>5:
				Y.append([1,0])
			else:
				Y.append([0,1])
	X=numpy.array(X)
	Y=numpy.array(Y)
	trX=X[0:320]
	trY=Y[0:320]
	return trX, trY
