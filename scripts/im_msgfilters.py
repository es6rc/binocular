#!/usr/bin/env python

from __future__ import print_function
import roslib
roslib.load_manifest('binocular')
#import sys
import rospy
import cv2
#from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from track import *
import message_filters
import numpy as np

class image_converter:
    def __init__(self, sub_topic1, sub_topic2):
        self.im1_sub = message_filters.Subscriber(sub_topic1, Image)
        self.im2_sub = message_filters.Subscriber(sub_topic2, Image)
        
        self.bridge = CvBridge()

        self.detected1 = False
        self.detected2 = False
        self.ctr1 = None
        self.ctr2 = None

        self.ts = message_filters.TimeSynchronizer([self.im1_sub, self.im2_sub], 10)
        self.ts.registerCallback(self.callback)
    
    def callback(self, im1, im2):
        try:
            cv_im1 = self.bridge.imgmsg_to_cv2(im1, "bgr8")
            cv_im2 = self.bridge.imgmsg_to_cv2(im2, "bgr8")
        except  CvBridgeError as e:
            print(e)
        
        self.detected1, self.ctr1, im1_track = track(cv_im1)
        self.detected2, self.ctr2, im2_track = track(cv_im2)
        #v_concat = np.concatenate((im1_track, im2_track), axis=0)
        #cv2.imshow('tracking', v_concat)
        #cv2.waitKey(1)
        #cv2.imshow('im2 tracking', im2_track)
        #cv2.waitKey(1)
        

    
 