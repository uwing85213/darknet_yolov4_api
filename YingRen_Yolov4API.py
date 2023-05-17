# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:39:29 2020

@author: MH-Lin
"""


import numpy as np
#yolov4
from ctypes import *
import random
import os
import cv2
import time
#from yolov4API import darknet #重要
import darknet #重要
#
import sys
sys.path.append("..")
#from utils import label_map_util
# import utils.yolov4_label_util as yolo_label
import random
#必要參數

#

class Yolov4YingRen():

    def __init__(self, config_file_Path, data_file_path,weights_Path,gpuid=0,thresh=0.25):
        #cfg路徑 data路徑 weight路徑 信心值
        
        #self.names, CLASS_NUM = yolo_label.load_names(PRED_NAMES)#回傳label and number
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
        if type(img) == str:  # 給自串路徑
        #    img = cv2.imread(img)
            image = img.copy()
        #    img1 = cv2.resize(img, (608, 608))
        #    frame = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
        elif type(img) == np.ndarray:  # 已經是圖片
            image = img.copy()
        #    img1 = cv2.resize(img.copy(), (608, 608))
        #    frame = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
        else:
            print('錯誤，Please input image or image path.')
        
        #要取得boxes, scores, labels
        #前置處理
        frame=image.copy()#複製來源圖片
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#轉成RGB
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height),interpolation=cv2.INTER_LINEAR)        
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())#影像轉成darknet專屬格式
        #inference 辨識
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image,min_score_thresh)
        
        objs=[]
        w=img.shape[1]
        h=img.shape[0]
        for label, confidence, bbox in detections:
            obj = []  # 每次清空
            left, top, right, bottom = darknet.bbox2points(bbox)
            
            xmin = int((left/self.width)*w)
            ymin = int((top/self.height)*h)
            xmax = int((right/self.width)*w)
            ymax = int((bottom/self.height)*h)
            #
            obj.append(label) # label name
            obj.append(np.array([xmin, ymin, xmax, ymax])) # 座標
            obj.append(np.array([xmin, ymin,xmax - xmin, ymax - ymin])) # w, h
            obj.append(float(confidence)) # 信心值
            objs.append(obj)            
        
        
        if visualize_boxes_IO == 'ON':  # 判斷FastRCNN內建可試化視窗開啟或關閉
            #image = darknet.draw_boxes(detections, frame_resized, self.class_colors)
            for label, confidence, bbox in detections:
                left, top, right, bottom = darknet.bbox2points(bbox)
                #print('A',bbox)
                xmin = int((left/self.width)*w)
                ymin = int((top/self.height)*h)
                xmax = int((right/self.width)*w)
                ymax = int((bottom/self.height)*h)
                #print('B',left,top,right,bottom)
                cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), self.class_colors[label], bbsize)
                cv2.putText(frame_rgb, "{} {:.2f}%".format(label, float(confidence)),
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fsize, self.class_colors[label], 2)
            image=frame_rgb.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        #mage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        
        return image, objs 
        