from conv_net import *

import numpy as np
import cv2
import glob
import os

DATA_PATH = './face/'
params = './params/gender_5x5_5_5x5_10.param'

if __name__ == '__main__':

    files = glob.glob(os.path.join(DATA_PATH, '*.*'))

    # This is the configuration of the full convolutional part of the CNN
    # `d` is a list of dicts, where each dict represents a convolution-maxpooling
    # layer. 
    # Eg c1 - first layer, convolution window size
    # p1 - first layer pooling window size
    # f_in1 - first layer no. of input feature arrays
    # f_out1 - first layer no. of output feature arrays
    d = [{'c1':(5,5),
          'p1':(2,2),
          'f_in1':1, 'f_out1':5},
         {'c2':(5,5),
          'p2':(2,2),
          'f_in2':5, 'f_out2':10}]

    # This is the configuration of the mlp part of the CNN
    # first tuple has the fan_in and fan_out of the input layer
    # of the mlp and so on.
    nnet =  [(800,256),(256,2)]

    c = ConvNet(d,nnet, (45,45))
    c.load_params(params)
    
    # loop through the fils
    # '0' is for female
    # '1' f for male
    for f in files:
        img = cv2.imread(f, 0)
        img = cv2.equalizeHist(img)
        
        resized_img = cv2.resize(img, (45,45))/255.

        out = np.argmax(c.predict(np.array([resized_img])))

        res = 'Male' if out == 1 else 'Female'
        print 'Prediction : {}'.format(res)

        cv2.imshow('Img', img)
        cv2.waitKey()
