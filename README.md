# DeepAestheticLearning
Authors- Anhad Mohananey, Yashovardhan Chaturvedi

Convolutional Neural Networks to predict aesthetic goodness of an image.

To train the CNN:

import build_model
predictor = build_model.trainly(<path_of_images_folder_for_training>, <no_of_epochs>)

The predictor object stores the trained model, and can be used for predicting whether a particular image is aesthetic or not, like so:

import cv2

res_image=cv2.imread(<path_of_test_ime>).reshape(-1,3,250,250)
print predictor(es_image)
The above will print a class label, (Aesthetic/Not Aesthetic)

Note: The 'factored' folder consists of 250X250X3 images scraped from the AVA/imagechallenge dataset(http://www.lucamarchesotti.com), which can be used for training purposes. The images that were originally downloaded were cropped to bring to 250*250*3. I would recommend downloading more images, particularly if you have a GPU, and are not constrained by training time.

Dependencies: Theano, Numpy, urllib, cv2(opencv in python).  

scrap.py has the code for the scrapper that I used to get data, and consequently crop and process it to 250*250*3 px. You can use it as a reference or write your own. In case this in violation of AVA's guidelines, please let me know and I will remove it.
