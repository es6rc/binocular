#!/usr/bin/env python

import cv2
import numpy as np


def cali_in():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

    # open the camera
    vd1 = cv2.VideoCapture(1)
    # vd2 = cv2.VideoCapture(2)

    #if vd1.isOpened():  # and vd2.isOpened(): # try to get the first frame
    rval1, frame1 = vd1.read()
    # rval2, frame2 = vd2.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #else:
    #    rval1 = False
        # rval2 = False

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Initial gray
    for i in range(10):

        rval1, frame1 = vd1.read()
        if rval1:
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            print gray.shape
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(frame1, (7, 5), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1000)
                print i
            else:
                i -= 1
    # while rval1 and rval2:
        # rval2, frame2 = vd2.read()
        # cv2.imshow("cap1", frame1)
        # cv2.imshow("cap2", frame2)
        # key = cv2.waitKey(30)
        # if key == 27:  # exit on ESC
        #     break
    cv2.destroyAllWindows()
    vd1.release()
    # vd2.release()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    R = cv2.Rodrigues(rvecs[0])[0]

    print ("K\n", mtx)
    print ("D\n", dist)
    print ("R\n", R)
    M = np.hstack((R, tvecs[0]))
    print ("M\n", M)
    P = np.matmul(mtx, M)
    print ("P\n", P)


def get_ex_cali_imgs():
    vd1 = cv2.VideoCapture(1)
    vd2 = cv2.VideoCapture(2)
    if vd1.isOpened() and vd2.isOpened():
        rval1, frame1 = vd1.read()
        rval2, frame2 = vd2.read()
    else:
        rval1 = False
        rval2 = False
    if rval1 and rval2:
        cv2.imshow("cap1", frame1)
        cv2.imshow("cap2", frame2)

        key = cv2.waitKey(0)
        if key == 27:  # exit on ESC
            cv2.imwrite('left_img.jpg', frame1)
            cv2.imwrite('right_img.png', frame2)
    cv2.destroyAllWindows()
    vd1.release()
    vd2.release()

def cali_ex():
    img1 = cv2.imread('left_img.jpg', 0)  # queryimage # left image
    img2 = cv2.imread('right_img.png', 0)  # trainimage # right image

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)


    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)


    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

cali_ex()