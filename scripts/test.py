#!/usr/bin/env python


# from os import listdir
from os.path import join
from os import walk
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def orbdetect(img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow('orb', img2)  # , plt.show()
    cv2.waitKey()


def harris(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray1 = np.float32(gray)
    dst = cv2.cornerHarris(gray1, 2, 3, 0.04)
    # a = [dst > 0.2 * dst.max()]
    # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, .2 * dst.max(), 255, 0)
    # corners2 = cv2.cornerSubPix(gray, dst, (11, 11), (-1, -1), criteria)
    nz = np.nonzero(dst)
    corners = np.float32(np.array([[[row, col]] for row, col in zip(nz[0], nz[1])]))
    cornerstempe = corners
    corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0] = [0, 0, 255]
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return corners2


def findchessboardcors(img):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img1 = img
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    ret, cors = cv2.findChessboardCorners(gray, (7, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(gray, cors, (11, 11), (-1, -1), criteria)#.reshape(-1,2)
        cv2.drawChessboardCorners(img1, (7, 7), corners2, ret)
        cv2.imshow('img', img1)
        cv2.waitKey()
        a = corners2[0][0][0]
        b = corners2[1][0][0]
        # if abs(corners2[0][0][0] - corners2[1][0][0]) > 0.1:
        #     corsa = np.reshape(corners2, (7, 7, 2))
        return corners2
    return None


def cali(fimgs):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in fimgs:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def mtch(img1, img2):
    # Initiate ORB detector
    img1_ = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_ = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1_, None)
    kp2, des2 = orb.detectAndCompute(img2_, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1_, kp1, img2_, kp2, matches[:10], None, flags=2)
    plt.imshow(img3), plt.show()


def goodft(img):
    img_ = img
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 70, 0.005, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img_, (x, y), 3, 255, -1)

    plt.imshow(img_), plt.show()


def stereo(imgL, imgR):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'gray')
    plt.show()


def test1():
    img_ = cv2.imread('/home/z/catkin_ws/src/binocular/scripts/parallelstereo/l_2.0.jpg', 0)
    # corners_1 = harris(img_)
    # cali(img_)
    img_1 = cv2.imread('/home/z/catkin_ws/src/binocular/scripts/parallelstereo/r_2.0.jpg', 0)
    # corners_2 = harris(img_1)
    # mtch(img_, img_1)
    stereo(img_, img_1)
    # goodft(img_)


def getLRimglists(direc='/home/z/catkin_ws/src/binocular/scripts/vergencestereo/'):
    f1 = []
    for (dirpath, dirnames, filenames) in walk(direc):
        # Find all image path in the dirpath
        f1 = [join(dirpath, filename) for filename in filenames]
        # f1.extend(filenames)
        break
    # Sort the list with its numbers in image path
    f1.sort(key=lambda f: float(filter(str.isdigit, f)))
    a = f1[1][-9:-8]
    # find all image paths for left cameras
    fl = [fi for fi in f1 if fi[-9:-8] == 'l']
    fl_chessboard_detectable = fl[3:21]
    fr = [fi for fi in f1 if fi[-9:-8] == 'r']
    fr_chessboard_detectable = fr[3:7]
    return fl_chessboard_detectable, fr_chessboard_detectable


def getLRmtxs(fl_chessboard_detectable, fr_chessboard_detectable):
    # Calibrate two cameras
    retl, mtxl, distl, rvecsl, tvecsl = cali(fl_chessboard_detectable)
    retr, mtxr, distr, rvecsr, tvecsr = cali(fr_chessboard_detectable)

    # Find rotation matrices and translation matrices
    # Rl = cv2.Rodrigues(rvecsl[0])[0]
    # Rr = cv2.Rodrigues(rvecsr[0])[0]
    # tl = tvecsl[0]
    # tr = tvecsr[0]
    # Find Projection matrix = mtrx*[R|t]
    # Pl = np.matmul(mtxl, np.concatenate((Rl, tl), axis=1))
    # Pr = np.matmul(mtxr, np.concatenate((Rr, tr), axis=1))
    return mtxl, mtxr


def trianpoints(cnrsl, cnrsr, kl, kr):
    pts_l_norm = cv2.undistortPoints(cnrsl, cameraMatrix=kl, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(cnrsr, cameraMatrix=kr, distCoeffs=None)
    E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999,
                                   threshold=3.0)
    points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

    mr = np.hstack((R, t))
    ml = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    P_l = np.dot(kl, ml)
    P_r = np.dot(kr, mr)
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, cnrsl, cnrsr)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :]
    return point_3d


if __name__ == '__main__':
    fl, fr = getLRimglists()
    # kl, kr = getLRmtxs(fl, fr)
    kl = np.array([[610.1799470098168, 0.0, 512.5],
                   [0.0, 610.1799470098168, 384.5],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    kr = np.array([[610.1799470098168, 0.0, 512.5],
                   [0.0, 610.1799470098168, 384.5],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    imgl = cv2.imread(fl[0])
    imgr = cv2.imread(fr[0])
    cnrsl = findchessboardcors(imgl)
    cnrsr = findchessboardcors(imgr)
    pts3d = trianpoints(cnrsl, cnrsr, kl, kr)
    coors_x = np.mgrid[0:7, 0:7][0].reshape(-1).astype(np.float32)[0:6]
    coors_y = np.mgrid[0:7, 0:7][1].reshape(-1).astype(np.float32)[0:6]
    x_grid, y_grid = np.meshgrid(coors_x, coors_y)
    coors_z = pts3d[2]#[0:49:7]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.plot_surface(x_grid, y_grid, coors_z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.scatter3D(coors_x, coors_y, coors_z, c=coors_z, cmap='Reds')
    plt.show()
