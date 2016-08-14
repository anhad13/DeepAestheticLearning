import urllib
import os
file=open("AVA.txt","r")
lines=file.readlines()
for line in lines:
	arr=line[:-2].split(' ')
	id1=arr[1]
	idx=arr[14]
	if os.path.exists("original/"+id1+".jpg")==False:
		url="http://images.dpchallenge.com/images_challenge/1000-1999/"+idx+"/1200/Copyrighted_Image_Reuse_Prohibited_"+id1+".jpg"
		print url
		res=urllib.urlopen(url)
		output = open("original/"+id1+".jpg","wb")
		output.write(res.read())
		output.close()
#this is a scraper that I used for scraping and resizing/cropping images appropriately.
import os
import cv2
import numpy
rootdir = 'original/'
for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		print file
		img=cv2.imread(rootdir+file)
		if img==None:
			break
		d1=img.shape[0]
		d2=img.shape[1]
		if d1>d2:
			#we have to crop d1 eventually
			resize_factor=d2/250.0
			resized=cv2.resize(img, (250,int(d1/resize_factor)))
			new_d1=resized.shape[0]
			to_cut=int((new_d1-250)/2)
			f=resized[to_cut:new_d1-to_cut,:]
			f=f[0:250,:]
		else:
			#we have to crop d2 eventually
			resize_factor=d1/250.0
			resized=cv2.resize(img, (int(d2/resize_factor),250))
			new_d2=resized.shape[0]
			to_cut=int((new_d2-250)/2)
			f=resized[:,to_cut:new_d2-to_cut]
			f=f[:,0:250]
		cv2.imwrite('factored/'+file,f)
