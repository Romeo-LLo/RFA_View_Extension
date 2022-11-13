import cv2
import glob
import os
import cv2.aruco as aruco
import numpy as np
import random
from needle_utils_temp import *

mtx, dist = camera_para_retrieve()
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
squareLength = 1.3
markerLength = 0.9
arucoParams = aruco.DetectorParameters_create()

cap = cv2.VideoCapture('../All_images/needle_trans3.mp4')
count = 0
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)

        if corners:
            diamondCorners, diamondIds = aruco.detectCharucoDiamond(image, corners, ids, squareLength / markerLength)
            if diamondCorners:
                cv2.imwrite("../All_images/offset3/frame%d.jpg" % count, image)  # save frame as JPG file
                print(count)
                count += 1
cap.release()
