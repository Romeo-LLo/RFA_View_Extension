import cv2
import glob
import os
import cv2.aruco as aruco
import numpy as np
import random
from needle_utils_temp import *

def offset_ransac():
    mtx, dist = camera_para_retrieve()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.3
    markerLength = 0.9
    arucoParams = aruco.DetectorParameters_create()
    imgs = glob.glob(os.path.join("../All_images/offset", "*.jpg"))


    for i in range(10):
        samples = random.sample(range(0, len(imgs)), 2)
        sample1 = cv2.imread(imgs[samples[0]])
        sample2 = cv2.imread(imgs[samples[1]])
        # cv2.imshow('1', sample1)
        # cv2.imshow('2', sample2)
        # cv2.waitKey(0)
        corners1, ids1, rejectedImgPoints = aruco.detectMarkers(sample1, aruco_dict, parameters=arucoParams)
        corners2, ids2, rejectedImgPoints = aruco.detectMarkers(sample2, aruco_dict, parameters=arucoParams)

        if np.any(ids1 != None) and np.any(ids2 != None):
            diamondCorners1, diamondIds1 = aruco.detectCharucoDiamond(sample1, corners1, ids1, squareLength / markerLength)
            diamondCorners2, diamondIds2 = aruco.detectCharucoDiamond(sample2, corners2, ids2, squareLength / markerLength)

            if diamondCorners1 and diamondCorners2:  # if aruco marker detected
                rvec1, tvec1, _ = aruco.estimatePoseSingleMarkers(diamondCorners1, squareLength, mtx, dist)
                rvec2, tvec2, _ = aruco.estimatePoseSingleMarkers(diamondCorners2, squareLength, mtx, dist)

                r_matrix1, _ = cv2.Rodrigues(rvec1[0][0])
                r_matrix2, _ = cv2.Rodrigues(rvec2[0][0])
                r_matrix1 *= -1
                r_matrix2 *= -1

                inv_r1 = np.linalg.inv(r_matrix1)
                B = np.matmul(inv_r1, tvec2[0][0] - tvec1[0][0])
                A = np.identity(3) - np.matmul(inv_r1, r_matrix2)
                offset = np.linalg.lstsq(A, B, rcond=None)[0]
                print(offset)
                trans1 = np.matmul(offset, r_matrix1.T)
                needle_tvec1 = tvec1[0][0] + trans1.T
                trans2 = np.matmul(offset, r_matrix2.T)
                needle_tvec2 = tvec2[0][0] + trans2.T

                # print(needle_tvec1)
                # print(needle_tvec2)
                print(np.linalg.norm(needle_tvec1-needle_tvec2))
                offset = np.array([0, 18.5, 2.5])

                trans1 = np.matmul(offset, r_matrix1.T)
                needle_tvec1 = tvec1[0][0] + trans1.T
                trans2 = np.matmul(offset, r_matrix2.T)
                needle_tvec2 = tvec2[0][0] + trans2.T
                # print(needle_tvec1)
                # print(needle_tvec2)
                print(np.linalg.norm(needle_tvec1-needle_tvec2))

                print('-------------------------')


def offset_ransac_concat():
    # result [1.07243335 20.38937994  2.73006118]

    mtx, dist = camera_para_retrieve()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.3
    markerLength = 0.9
    arucoParams = aruco.DetectorParameters_create()
    imgs = glob.glob(os.path.join("../All_images/offset2", "*.jpg"))

    r1 = np.array([]).reshape(0, 3)
    r2 = np.array([]).reshape(0, 3)
    t1 = np.array([])
    t2 = np.array([])

    for i in range(100):
    # for i in range(len(imgs)-1):
        samples = random.sample(range(0, len(imgs)), 2)
        sample1 = cv2.imread(imgs[samples[0]])
        sample2 = cv2.imread(imgs[samples[1]])
        # sample1 = cv2.imread(imgs[i])
        # sample2 = cv2.imread(imgs[i+1])

        corners1, ids1, rejectedImgPoints = aruco.detectMarkers(sample1, aruco_dict, parameters=arucoParams)
        corners2, ids2, rejectedImgPoints = aruco.detectMarkers(sample2, aruco_dict, parameters=arucoParams)

        if np.any(ids1 != None) and np.any(ids2 != None):
            diamondCorners1, diamondIds1 = aruco.detectCharucoDiamond(sample1, corners1, ids1, squareLength / markerLength)
            diamondCorners2, diamondIds2 = aruco.detectCharucoDiamond(sample2, corners2, ids2, squareLength / markerLength)

            if diamondCorners1 and diamondCorners2:  # if aruco marker detected
                rvec1, tvec1, _ = aruco.estimatePoseSingleMarkers(diamondCorners1, squareLength, mtx, dist)
                rvec2, tvec2, _ = aruco.estimatePoseSingleMarkers(diamondCorners2, squareLength, mtx, dist)

                r_matrix1, _ = cv2.Rodrigues(rvec1[0][0])
                r_matrix2, _ = cv2.Rodrigues(rvec2[0][0])

                r1 = np.vstack([r1, r_matrix1])
                r2 = np.vstack([r2, r_matrix2])

                t1 = np.hstack([t1, tvec1[0][0]])
                t2 = np.hstack([t2, tvec2[0][0]])


    r = r1 - r2
    t = t2 - t1
    offset = np.linalg.lstsq(r, t, rcond=None)[0]
    print(offset)


    # inv_r1 = np.linalg.inv(r1)
    # B = np.matmul(inv_r1, t2 - t1)
    # A = np.identity(3) - np.matmul(inv_r1, r2)
    # offset = np.linalg.lstsq(A, B, rcond=None)[0]



def offset_ransac_3D_known_grid():

    # [ -0.19394971 -20.76504864  -3.83135322]
    mtx, dist = camera_para_retrieve()
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.3
    markerLength = 0.9
    arucoParams = aruco.DetectorParameters_create()
    imgs = glob.glob(os.path.join("../All_images/offset_tip1129", "*.jpg"))

    r1 = np.array([]).reshape(0, 3)
    t1 = np.array([])
    t2 = np.array([])


    board_coordinate = np.load("../Coordinate/board_coordinate.npy")
    fix_pt = board_coordinate[0]
    fix_pt = fix_pt.T
    for i in range(len(imgs)):
    # for i in range(300):

        sample1 = cv2.imread(imgs[i])
        corners1, ids1, rejectedImgPoints = aruco.detectMarkers(sample1, aruco_dict, parameters=arucoParams)

        if np.any(ids1 != None):
            diamondCorners1, diamondIds1 = aruco.detectCharucoDiamond(sample1, corners1, ids1, squareLength / markerLength)

            if diamondCorners1:  # if aruco marker detected
                rvec1, tvec1, _ = aruco.estimatePoseSingleMarkers(diamondCorners1, squareLength, mtx, dist)

                r_matrix1, _ = cv2.Rodrigues(rvec1[0][0])

                r1 = np.vstack([r1, r_matrix1])
                t1 = np.hstack([t1, tvec1[0][0]])
                t2 = np.hstack([t2, fix_pt])


    r = r1
    t = t2 - t1
    offset = np.linalg.lstsq(r, t, rcond=None)[0]

    print(offset)


def offset_ransac_3D_known_grid_original_tool():

    # [ -0.19394971 -20.76504864  -3.83135322]
    mtx, dist = camera_para_retrieve()
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.3
    markerLength = 0.9
    arucoParams = aruco.DetectorParameters_create()
    imgs = glob.glob(os.path.join("../All_images/offset_original_tool", "*.jpg"))

    r1 = np.array([]).reshape(0, 3)
    t1 = np.array([])
    t2 = np.array([])


    board_coordinate = np.load("../Coordinate/board_coordinate.npy")
    fix_pt = board_coordinate[1]
    fix_pt = fix_pt.T
    for i in range(len(imgs)):

        sample1 = cv2.imread(imgs[i])
        corners1, ids1, rejectedImgPoints = aruco.detectMarkers(sample1, aruco_dict, parameters=arucoParams)

        if np.any(ids1 != None):
            diamondCorners1, diamondIds1 = aruco.detectCharucoDiamond(sample1, corners1, ids1, squareLength / markerLength)

            if diamondCorners1:  # if aruco marker detected
                rvec1, tvec1, _ = aruco.estimatePoseSingleMarkers(diamondCorners1, squareLength, mtx, dist)

                r_matrix1, _ = cv2.Rodrigues(rvec1[0][0])

                r1 = np.vstack([r1, r_matrix1])
                t1 = np.hstack([t1, tvec1[0][0]])
                t2 = np.hstack([t2, fix_pt])


    r = r1
    t = t2 - t1
    offset = np.linalg.lstsq(r, t, rcond=None)[0]

    print(offset)







if __name__ == "__main__":
    # offset_ransac_concat()
    # offset_ransac_3D_known_grid()
    offset_ransac_3D_known_grid_original_tool()