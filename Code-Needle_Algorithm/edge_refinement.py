
import os
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import argrelextrema, find_peaks
import bisect
import cv2.aruco as aruco
from bresenham import bresenham
from scipy import odr
import cv2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import TIS
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from needle_utils import *
from Linear_equation import *
import torch
import gi
import cv2.aruco as aruco
import datetime
from matplotlib import pyplot as plt

gi.require_version("Gst", "1.0")
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_final4.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)

def detect():
    path = '../All_images/edge_investigate/Nolables'
    images = glob.glob('../All_images/edge_investigate/Nolables/*.jpg')

    # frame = cv2.imread('../All_images/edge_investigate/Nolables/32-45-12.24.jpg')

    for img in images:
        frame = cv2.imread(img)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            for i in range(1, 11):
                cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 255, 0), -1)

                rf_x, rf_y = edge_refinement(gray_frame, x[i], y[i], i)
                # print(x[i], y[i], rf_x, rf_y)
                cv2.circle(frame, (rf_x, rf_y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (rf_x, rf_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

        # cv2.imshow('Window', frame)
        # cv2.waitKey(0)


def edge_refinement(gray_frame, x, y, id):

    region_size = 6
    h, w = gray_frame.shape
    # kernel = np.concatenate((np.full((3, 5), -1), np.full((2, 5), 1)), axis=0)
    if id % 2 == 0:
        # kernel = np.array([[1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1],
        #                    [0, 0, 0, 0, 0],
        #                    [-1, -1, -1, -1, -1],
        #                    [-1, -1, -1, -1, -1]])

        # kernel = np.array([[ 0,  1,  1,  1,  1],
        #                    [-1,  0,  1,  1,  1],
        #                    [-1, -1,  0,  1,  1],
        #                    [-1, -1, -1,  0,  1],
        #                    [-1, -1, -1, -1,  0]])

        kernel = np.array([[1, 1,   1,  1,  0],
                           [1, 1,   1,  0, -1],
                           [1, 1,   0, -1, -1],
                           [1, 0,  -1, -1, -1],
                           [0, -1, -1, -1, -1]])

    else:
        # kernel = np.array([[-1, -1, -1, -1, -1],
        #                    [-1, -1, -1, -1, -1],
        #                    [0, 0, 0, 0, 0],
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1]])
        # kernel = np.array([[0, -1, -1, -1, -1],
        #                    [1,  0, -1, -1, -1],
        #                    [1,  1,  0, -1, -1],
        #                    [1,  1,  1,  0, -1],
        #                    [1,  1,  1,  1,  0]])

        kernel = np.array([[-1, -1, -1, -1,  0],
                           [-1, -1, -1,  0,  1],
                           [-1, -1,  0,  1,  1],
                           [-1,  0,  1,  1,  1],
                           [0,   1,  1,  1,  1]])


    tlx = max(round(x - region_size), 0)
    tly = max(round(y - region_size), 0)

    lrx = min(round(x + region_size), w)
    lry = min(round(y + region_size), h)
    region = gray_frame[tly: lry, tlx: lrx]


    conv = convolve2D(region, kernel)


    # plt.subplot(1, 3, 1), plt.imshow(region, cmap='gray')
    # plt.title('Region'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 3, 2), plt.imshow(conv, cmap='gray')
    # plt.title('Conv'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()
    reg_max = np.unravel_index(np.argmax(conv, axis=None), conv.shape)

    cv2.circle(region, (reg_max[1]+2, reg_max[0]+2), 1, (0, 255, 0), -1)
    #
    cv2.imshow('123', region)
    cv2.waitKey(0)
    rf_x = round(x - region_size + reg_max[1] + 2)
    rf_y = round(y - region_size + reg_max[0] + 2)
    return rf_x, rf_y

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    # kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[1]
    yKernShape = kernel.shape[0]
    xImgShape = image.shape[1]
    yImgShape = image.shape[0]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((yOutput, xOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[1] + padding*2, image.shape[0] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[0]):
        # Exit Convolution
        if y > image.shape[0] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[1]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[1] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[y, x] = (kernel * imagePadded[y: y + yKernShape, x: x + xKernShape]).sum()
                except:
                    break

    return output

#
# def convolve2D(image, kernel, padding=2, strides=1):
#     # Cross Correlation
#     # kernel = np.flipud(np.fliplr(kernel))
#
#     # Gather Shapes of Kernel + Image + Padding
#     xKernShape = kernel.shape[0]
#     yKernShape = kernel.shape[1]
#     xImgShape = image.shape[0]
#     yImgShape = image.shape[1]
#
#     # Shape of Output Convolution
#     xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
#     yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
#     output = np.zeros((xOutput, yOutput))
#
#     # Apply Equal Padding to All Sides
#     if padding != 0:
#         imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
#         imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
#         print(imagePadded)
#     else:
#         imagePadded = image
#
#     # Iterate through image
#     for y in range(image.shape[1]):
#         # Exit Convolution
#         if y > image.shape[1] - yKernShape:
#             break
#         # Only Convolve if y has gone down by the specified Strides
#         if y % strides == 0:
#             for x in range(image.shape[0]):
#                 # Go to next row once kernel is out of bounds
#                 if x > image.shape[0] - xKernShape:
#                     break
#                 try:
#                     # Only Convolve if x has moved by the specified Strides
#                     if x % strides == 0:
#                         output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
#                 except:
#                     break
#
#     return output
if __name__ == "__main__":
    detect()