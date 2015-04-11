from conv_net import *

import numpy as np
import cv2
import glob
import os

DATA_PATH = './face/'
cnn_params = './params/gender_big.param'
face_params = './params/lbpcascade_frontalface.xml'
#face_params = './params/haarcascade_frontalface_alt.xml'

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
          'f_in1':1, 'f_out1':20},
         {'c2':(5,5),
          'p2':(2,2),
          'f_in2':20, 'f_out2':100}]

    # This is the configuration of the mlp part of the CNN
    # first tuple has the fan_in and fan_out of the input layer
    # of the mlp and so on.
    nnet =  [(6400,512),(512,2)]

    c = ConvNet(d,nnet, (45,45))
    c.load_params(cnn_params)
    
    face_cascade = cv2.CascadeClassifier(face_params)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    while(True):
        val, image = cap.read()

        image = cv2.pyrDown(image, 0.15)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,255))

            face_image = gray[y:y+h, x:x+w]
            
            resized_img = cv2.resize(face_image, (45,45))/255.

            out = np.argmax(c.predict(np.array([resized_img])))
            
            res = 'Male' if out == 1 else 'Female'
            cv2.putText(image, res, (max(x-10,0), max(y-10, 0)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, 255)
            
            #print 'Prediction : {}'.format(res)

        cv2.imshow('Image', image)
        cv2.waitKey(10)
