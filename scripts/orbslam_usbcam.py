#!/usr/bin/env python

import roslib
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# open the camera
vd1 = cv2.VideoCapture(1)
vd2 = cv2.VideoCapture(2)
bridge = CvBridge()
if vd1.isOpened() and vd2.isOpened():
    rval1, frame1 = vd1.read()
    rval2, frame2 = vd2.read()
else:
    rval1 = False
    rval2 = False
rospy.init_node('camera', anonymous=True)
left_pub = rospy.Publisher("camera/left/image_raw", Image, queue_size=10)
right_pub = rospy.Publisher("camera/right/image_raw", Image, queue_size=10)

while rval1 and rval2:
    rval1, frame1 = vd1.read()
    rval2, frame2 = vd2.read()
    try:
        left_pub.publish(bridge.cv2_to_imgmsg(frame1, "bgr8"))
        right_pub.publish(bridge.cv2_to_imgmsg(frame2, "bgr8"))
    except CvBridgeError as e:
        print(e)
    cv2.imshow("left", frame1)
    cv2.imshow("right", frame2)
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        break