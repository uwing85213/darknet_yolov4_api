# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:39:29 2020

@author: MH-Lin
"""


import numpy as np
from ctypes import *
import random
import os
import cv2
import time
import darknet
import sys
sys.path.append("..")

import random
class Yolov4YingRen():

    def __init__(self, config_file_Path, data_file_path,weights_Path,gpuid=0,thresh=0.25):        
        darknet.setGPU_ID(gpuid)
        
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file_Path,
            data_file_path,
            weights_Path,
            batch_size=1
        )
    
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        self.colors = []
        for i in range(len(self.class_colors)):
            r = self.class_colors[self.class_names[i]][0]
            g = self.class_colors[self.class_names[i]][1]
            b = self.class_colors[self.class_names[i]][2]
            self.colors.append((b, g, r))
    
    
    def catchObject_J(self, img, min_score_thresh, visualize_boxes_IO="OFF",bbsize=1,fsize=0.5):#回傳image 物件list
        if type(img) == np.ndarray or type(img) == str:
            image = img.copy()
        else:
            print('Error ，Please input image or image path.')
        
        #boxes, scores, labels
        frame=image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height),interpolation=cv2.INTER_LINEAR)        
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        #inference
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image,min_score_thresh)
        
        objs=[]
        w=img.shape[1]
        h=img.shape[0]
        for label, confidence, bbox in detections:
            obj = []
            left, top, right, bottom = darknet.bbox2points(bbox)
            
            xmin = int((left/self.width)*w)
            ymin = int((top/self.height)*h)
            xmax = int((right/self.width)*w)
            ymax = int((bottom/self.height)*h)
            #
            obj.append(label) # label name
            obj.append(np.array([xmin, ymin, xmax, ymax]))
            obj.append(np.array([xmin, ymin,xmax - xmin, ymax - ymin])) # w, h
            obj.append(float(confidence))
            objs.append(obj)            
                
        if visualize_boxes_IO == 'ON':
            for label, confidence, bbox in detections:
                left, top, right, bottom = darknet.bbox2points(bbox)
                xmin = int((left/self.width)*w)
                ymin = int((top/self.height)*h)
                xmax = int((right/self.width)*w)
                ymax = int((bottom/self.height)*h)
                cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), self.class_colors[label], bbsize)
                cv2.putText(frame_rgb, "{} {:.2f}%".format(label, float(confidence)),
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fsize, self.class_colors[label], 2)
            image=frame_rgb.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return image, objs 
        