#!/usr/bin/env python
# coding: utf-8

# In[7]:

#Written by Srikanth Paloji on 10.05.2021

import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
from playsound import playsound
from threading import Thread
import time


Frames = 10
count=0
alert_on = False

def play_s():
    playsound('alert.wav')

#Detects Face and shape of the Face with landmarks 
Face_detector = dlib.get_frontal_face_detector()
Face_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Defines Eye Aspect Ratio distances using euclidean distance from scipy_spacial
def Eye_aspect_ratio(a,b,c,d,e,f):
    A = distance.euclidean(b,d)
    B = distance.euclidean(c,d)
    C = distance.euclidean(a,f)
    ear = (A+B)/(2.0*C)
    return ear
#starts Video capturing from any standard camera
def main():
    capture = cv2.VideoCapture(0)
    
    while True:
        #ret is a boolean variable that returns true if the frame is available.
        #frame is an image array, captured based on the default frames per second 
        ret, frame = capture.read()
    
        #Considering gray image to analyse each frame for drowsiness detection.
        gray_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = Face_detector(gray_frames)
        
        for face in faces:
            landmarks = Face_landmarks(gray_frames, face)
        
            landmarks = face_utils.shape_to_np(landmarks) #converrting image to Numpy array
            lefteye_landmarks = Eye_aspect_ratio(landmarks[36],landmarks[37],landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            righteye_landmarks = Eye_aspect_ratio(landmarks[42],landmarks[43],landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            # creating landmarks on eyes 
            for n in range(36,48):
                (x,y) = landmarks[n]
                cv2.circle(frame, (x, y), 1, (255,255, 255), 1) # we use circle to create points as landmarks
            
            EAR = (lefteye_landmarks+righteye_landmarks)/2
            EAR = round(EAR,2)
            cv2.putText(frame,"EAR:{:.2f}".format(EAR),(300,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        
            if EAR<0.28:
                count+=1
                if count>=Frames:
                    if not alert_on:
                        alert_on =True
                        t=Thread(name="playsound", target=play_s)
                        t.deamon = True
                        t.start()
                cv2.putText(frame,"Feeling Drowsy?",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
            else:
                count = 0
                alert_on=False
            
        cv2.imshow("Drawsiness Detection",frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
if __name__=='__main__': # calling the main function 
    main()


# In[ ]:





# In[ ]:




