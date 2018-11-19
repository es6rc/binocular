#!/usr/bin/env python

'''
Track a green ball using OpenCV.
    Copyright (C) 2015 Conan Zhao and Simon D. Levy
'''

import cv2
import numpy as np

#WINDOW_NAME = 'BallTracker'

def track(image):
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5,5), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Threshold the HSV im for only green colors
    lower_green = np.array([29, 86, 6])
    upper_green = np.array([64, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Blur the mask
    bmask = cv2.GaussianBlur(mask, (5,5), 0)

    # Take the moments to get the centroid
    moments = cv2.moments(bmask)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None
    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)


    '''
    These Lines find contour of the ball

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    '''
    # Assume no centroid
    ctr = (-1,-1)

    # Set a Boolean value to tell if the ball has been detected or not:
    detected = False
    # Use centroid if it exists
    if centroid_x != None and centroid_y != None:

        ctr = (centroid_x, centroid_y)
        detected = True
        # Put Red Circle at centroid of the ball in image
        cv2.circle(image, ctr, 4, (0,0,255))

    # Display full-color image
    # cv2.imshow(window_name, image)
    # Force image display, setting centroid to None on ESC key input
    #if cv2.waitKey(1) & 0xFF == 27:
        #ctr = None
    
    # Return coordinates of centroid
    return detected, ctr, image

