# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 00:34:08 2022

@author: MH-Lin
"""
import cv2
import numpy as np
import os
import time 


from YingRen_Yolov4API import Yolov4YingRen


min_score_thresh = 0.5

PATH_TO_VIDEO='video/video.mp4'
video = cv2.VideoCapture(PATH_TO_VIDEO)
width1 = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

MODEL_PATH = 'model/coco/'
config_file_Path = MODEL_PATH + 'coco.cfg'
data_file_path = MODEL_PATH + 'coco.data'
weights_Path = MODEL_PATH + 'coco.weights'
detector = Yolov4YingRen(config_file_Path , data_file_path , weights_Path)

while(video.isOpened()):  
  starttime = time.time()
  ret, frame = video.read()

  if not ret:
      break
  img , objs = detector.catchObject_J(frame , min_score_thresh , 'ON',bbsize=1,fsize=0.5) 
  
  endtime = time.time()            
  frametime = endtime - starttime
  
  if frametime>=0:
         String =  'Fps:' + str(np.float16(1 / frametime))
         cv2.putText(img , String , (0,30) , cv2.FONT_HERSHEY_SIMPLEX, 1.2 , (0,0,255), thickness = 3) 
      

  cv2.imshow('windows',img)
  cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
