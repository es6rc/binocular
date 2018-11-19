#!/usr/bin/env python

from __future__ import print_function
import roslib
roslib.load_manifest('binocular')
import sys
import rospy
import cv2
#from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from track import *

class image_converter:

  def __init__(self, sub_topic, detecton):#, window_name):
    #self.image_pub = rospy.Publisher(pub_topic, Image)
    self.image_sub = rospy.Subscriber(sub_topic, Image, self.callback)

    self.bridge = CvBridge()
    
    self.detecton = detecton
    self.detected = False
    self.point_of_interest = None
    self.image_track = None
    self.raw_image = None
    #self.window_name = window_name

  def callback(self, data):
    try:
      self.raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

   
    #cv2.imshow("Image window", cv_image)
    
    #Image Processing -- Tracking
    if self.detecton:
      self.detected, self.point_of_interest, self.image_track = track(self.raw_image)
    #cv2.imshow(self.window_name, self.image_track)
    #cv2.waitKey(3)

    #try:
    #  self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.image_track, "bgr8"))
    #except CvBridgeError as e:
    #  print(e)
    '''
    def main(args):
      ic = image_converter('/binocular/righteye/image_raw')
      rospy.init_node('image_converter', anonymous=True)
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print("Shutting down")
      cv2.destroyAllWindows()

    if __name__ == '__main__':
        main(sys.argv)
    '''