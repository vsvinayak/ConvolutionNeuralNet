from conv_net import *

import numpy as np
import cv2
import glob
import os

DATA_PATH = './face/'
cnn_params = './params/gender_5x5_5_5x5_10.param'
#face_params = './params/lbpcascade_frontalface.xml'
face_params = './params/haarcascade_frontalface_alt.xml'

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
    c.load_params(cnn_params)
    
    face_cascade = cv2.CascadeClassifier(face_params)

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/vinayak/Videos/bw.mp4")

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    while(True):
        
        val, image = cap.read()
        
        if image is None:
            break
        #image = cv2.imread("./face/group_2.jpg")

        #image = cv2.pyrDown(image)
        image = cv2.resize(image, (320,240))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30,30))

        for f in faces:
            x,y,w,h = f
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,255))
            
            face_image = gray[y:y+h, x:x+w]

            resized_img = cv2.resize(face_image, (45,45))/255.
            
            pred = c.predict(np.array([resized_img]))
            out = np.argmax(pred)
            
            res = 'Male' if out == 1 else 'Female'
            cv2.putText(image, res, (max(x-10,0), max(y-10, 0)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, 255)
            
            
            print 'Prediction : {}'.format(pred)
        
        cv2.imshow('Image', image)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        #break
