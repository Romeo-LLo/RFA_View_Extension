
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
cfg.MODEL.WEIGHTS = '../Model_path/model_1201.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)

lazer = 1.036
plist = [1, 5, 7]
dlist = [60, 40]
dlist = [d / lazer for d in dlist]
tip_off = 2.2


def detect():
    path = '../All_images/edge_investigate/Nolables'
    images = glob.glob('../All_images/edge_investigate/Nolables/*.jpg')

    # frame = cv2.imread('../All_images/edge_investigate/Nolables/32-36-17.39.jpg')

    for img in images:
        frame = cv2.imread(img)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            m, b = line_polyfit(x, y)
            dx = x[1] - x[4]
            dy = y[1] - y[4]
            for i in range(1, 11):
                cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 255, 0), -1)
                kernel = kernel_choice(m, i, dx, dy)

                rf_x, rf_y = edge_refinement_conv(gray_frame, x[i], y[i], kernel)
                # print(x[i], y[i], rf_x, rf_y)
                cv2.circle(frame, (rf_x, rf_y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (rf_x, rf_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Window', frame)
            cv2.waitKey(0)



def detect_linear():
    mtx, dist = camera_para_retrieve()

    images = glob.glob('../All_images/error_investigate/different_keypoint_blank/*.jpg')

    for img in images:
        frame = cv2.imread(img)
        # frame = cv2.imread('../All_images/edge_investigate/Blank/50-13-11.78.jpg')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]


            if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):

                coord_3D_rf, coord_2D_rf = edge_refinement_linear_display(frame, gray_frame, x, y, plist)

                for i in range(3):
                    string = f"{coord_2D_rf[i][0]}, {coord_2D_rf[i][1]}"
                    cv2.putText(frame, string, (1000, 130 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 1, cv2.LINE_AA)
                tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0],
                                                            dlist[1], mtx, tip_off)

                error_rf = error_calc_board(tip_rf, anchor=0)
                error_rf = round(error_rf, 2)
                cv2.putText(frame, str(img), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(error_rf), (coord_2D_rf[2][0], coord_2D_rf[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                # for corner in coord_2D_rf:
                #     frame[corner[1]][corner[0]+1] = (0, 0, 255)

                cv2.imshow('Window', frame)
                cv2.waitKey(0)


def detect_linear_img():
    # ng_set = ['45-11', '44-50', '45-17', '44-56']
    ng_set = ['15-56', '15-19', '14-39', '14-18', '14-51']
    for num in ng_set:
        frame = cv2.imread(f"../All_images/TrainImage/{num}.jpg")
        # frame = cv2.imread(f"../All_images/Testimg_undist/{num}.jpg")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():
            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]



            if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):
                coord_3D_rf, coord_2D_rf = edge_db_refinement_linear_display(frame, gray_frame, x, y, list(range(9)))


                # for corner in coord_2D_rf:
                #     frame[corner[1]][corner[0]+1] = (0, 0, 255)

        cv2.imshow('Window', frame)
        cv2.waitKey(0)




def edge_refinement_linear_display(color_frame, gray_frame, x, y, plist):
    win = 10

    end_pts = end_pts_tip2end(x, y, gray_frame.shape)
    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    gray_pt_set = np.array(gray_pt_set)

    pixel = [gray_frame[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = np.gradient(pixel)

    coord_3D_rf = []
    coord_2D_rf = []
    for i in plist:
        rough_coord = (x[i], y[i])
        closest_index = find_closest_edge(gray_pt_set, rough_coord)
        pixel_inspect = pixel[closest_index - win: closest_index + win]
        deriv_inspect_o = d_pixel[closest_index - win: closest_index + win]
        deriv_inspect = deriv_inspect_o[1:-1]
        # note that window out of range
        norm_deriv = 255 * (deriv_inspect - np.min(deriv_inspect)) / (np.max(deriv_inspect) - np.min(deriv_inspect))

        if i % 2 == 0:
            peak_inspect = np.argmax(deriv_inspect)
        else:
            peak_inspect = np.argmin(deriv_inspect)

        mark = peak_inspect + 1
        refine_corner = gray_pt_set[closest_index - win + peak_inspect + 1]
        pt_rf = np.array([refine_corner[0], refine_corner[1], 0], dtype='float64')
        coord_3D_rf.append(pt_rf)
        coord_2D_rf.append(refine_corner)
        print(i, mark)
        print(pixel_inspect)
        print(deriv_inspect_o)

        k = 0
        for j in range(closest_index-win+1, closest_index+win-1):
            coord_x, coord_y = gray_pt_set[j]
            color_frame[coord_y][coord_x] = (0, 0, norm_deriv[k])
            k += 1

    return coord_3D_rf, coord_2D_rf



def edge_db_refinement_linear_display(color_frame, gray_frame, x, y, plist):
    win = 30

    end_pts = end_pts_tip2end(x, y, gray_frame.shape)
    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    gray_pt_set = np.array(gray_pt_set)

    pixel = [gray_frame[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = np.gradient(pixel)

    m, b = line_polyfit(x, y)
    dx = x[1] - x[4]
    dy = y[1] - y[4]

    coord_3D_rf = []
    coord_2D_rf = []
    for i in plist:
        rough_coord = (x[i], y[i])
        closest_index = find_closest_edge(gray_pt_set, rough_coord)
        pixel_inspect = pixel[closest_index - win: closest_index + win]
        deriv_inspect_o = d_pixel[closest_index - win: closest_index + win]
        deriv_inspect = deriv_inspect_o[1:-1]
        # note that window out of range
        norm_deriv = 255 * (deriv_inspect - np.min(deriv_inspect)) / (np.max(deriv_inspect) - np.min(deriv_inspect))

        fig, (ax, ax2) = plt.subplots(ncols=2, sharex=True)

        pt_index = np.linspace(1, deriv_inspect_o.shape[0], deriv_inspect_o.shape[0])
        ax.plot(pt_index, pixel_inspect, color='blue')
        size = 12
        # ax.set_xlabel('Points along the inspected segment', fontsize=size)
        ax.set_ylabel('Pixel values', fontsize=size)
        ax2.plot(pt_index, deriv_inspect_o, color='blue')
        # ax2.set_xlabel('Points along the inspected segment', fontsize=size)
        ax2.set_ylabel('Derivative of pixel values', fontsize=size)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax.set_box_aspect(1)
        ax2.set_box_aspect(1)
        # plt.xlabel("Points along the inspected segment")
        fig.text(0.5, 0.2, "Points along the inspected segment", ha='center', fontsize=size)

        plt.show()

        kernel = kernel_choice(m, i, dx, dy)
        top = 4
        if i % 2 == 0:
            indice = np.argpartition(deriv_inspect, -top)[-top:]
            peak = np.argmax(deriv_inspect)

        else:
            peak = np.argmin(deriv_inspect)
            deriv_inspect = np.negative(deriv_inspect)
            indice = np.argpartition(deriv_inspect, -top)[-top:]

        peak_inspect = peak
        max_prod = 0
        for idx in indice:
            pt = gray_pt_set[closest_index - win + idx + 1]
            region = cropArea(gray_frame, pt[0], pt[1])
            inner_product = (kernel * region).sum()
            if inner_product > max_prod:
                peak_inspect = idx
                max_prod = inner_product
        #         print(inner_product, idx)
        # print(f'original: {peak}, refined: {peak_inspect}')

        refine_corner = gray_pt_set[closest_index - win + peak_inspect + 1]
        pt_rf = np.array([refine_corner[0], refine_corner[1], 0], dtype='float64')
        coord_3D_rf.append(pt_rf)
        coord_2D_rf.append(refine_corner)


        k = 0
        for j in range(closest_index-win+1, closest_index+win-1):
            coord_x, coord_y = gray_pt_set[j]
            color_frame[coord_y][coord_x] = (0, 0, norm_deriv[k])
            k += 1

    return coord_3D_rf, coord_2D_rf




def edge_refinement_linear_mod2(gray_frame, x, y, plist):
    win = 10

    end_pts = end_pts_tip2end(x, y, gray_frame.shape)
    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    gray_pt_set = np.array(gray_pt_set)

    pixel = [gray_frame[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = np.gradient(pixel)

    m, b = line_polyfit(x, y)
    dx = x[1] - x[4]
    dy = y[1] - y[4]

    coord_3D_rf = []
    coord_2D_rf = []
    for i in plist:
        rough_coord = (x[i], y[i])
        closest_index = find_closest_edge(gray_pt_set, rough_coord)
        lower_bound = max(closest_index - win, 0)
        upper_bound = min(closest_index + win, len(gray_pt_set))

        deriv_inspect_o = d_pixel[lower_bound: upper_bound]
        deriv_inspect = deriv_inspect_o[1:-1]
        kernel = kernel_choice(m, i, dx, dy)
        top = 3
        if len(deriv_inspect) == 0:
            return None, None

        if i % 2 == 0:
            # initial guess with highest deriv
            peak = np.argmax(deriv_inspect)
            # check top few with template matching, extract indice of them
            indice = np.argpartition(deriv_inspect, -top)[-top:]

        else:
            peak = np.argmin(deriv_inspect)
            deriv_inspect = np.negative(deriv_inspect)
            indice = np.argpartition(deriv_inspect, -top)[-top:]

        # further refine with conv
        peak_inspect = peak
        max_prod = 0
        for idx in indice:
            pt = gray_pt_set[closest_index - win + idx + 1]
            region = cropArea(gray_frame, pt[0], pt[1])

            # on the boundary, use the max deriv, skip this point
            if region.shape != kernel.shape:
                break

            inner_product = (kernel * region).sum()
            if inner_product > max_prod:
                peak_inspect = idx
                max_prod = inner_product

        refine_corner = gray_pt_set[closest_index - win + peak_inspect + 1]
        pt_rf = np.array([refine_corner[0], refine_corner[1], 0], dtype='float64')
        coord_3D_rf.append(pt_rf)
        coord_2D_rf.append(refine_corner)



    return coord_3D_rf, coord_2D_rf


def edge_refinement_linear_mod(gray_frame, x, y, plist):
    win = 10

    end_pts = end_pts_tip2end(x, y, gray_frame.shape)
    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    gray_pt_set = np.array(gray_pt_set)

    pixel = [gray_frame[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = np.gradient(pixel)

    coord_3D_rf = []
    coord_2D_rf = []

    for i in plist:
        rough_coord = (x[i], y[i])
        closest_index = find_closest_edge(gray_pt_set, rough_coord)
        deriv_inspect = d_pixel[closest_index - win: closest_index + win]
        # remove first and last
        deriv_inspect = deriv_inspect[1:-1]

        if i % 2 == 0:
            peak_inspect = np.argmax(deriv_inspect)
        else:
            peak_inspect = np.argmin(deriv_inspect)
        #add 1
        refine_corner = gray_pt_set[closest_index - win + peak_inspect + 1]
        pt_rf = np.array([refine_corner[0], refine_corner[1], 0], dtype='float64')
        coord_3D_rf.append(pt_rf)
        coord_2D_rf.append(refine_corner)

    return coord_3D_rf, coord_2D_rf

def edge_refinement_linear(gray_frame, x, y, plist):
    win = 7
    m, b = np.polyfit(x, y, 1)
    end_pts = line_end_points_on_image(m, b, gray_frame.shape, True)
    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    gray_pt_set = np.array(gray_pt_set)

    pixel = [gray_frame[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = abs(np.gradient(pixel))

    coord_3D_rf = []
    coord_2D_rf = []

    for i in plist:
        rough_coord = (x[i], y[i])
        closest_index = find_closest_edge(gray_pt_set, rough_coord)
        deriv_inspect = d_pixel[closest_index - win: closest_index + win]

        peak_inspect = np.argmax(deriv_inspect)
        refine_corner = gray_pt_set[closest_index - win + peak_inspect]
        pt_rf = np.array([refine_corner[0], refine_corner[1], 0], dtype='float64')
        coord_3D_rf.append(pt_rf)
        coord_2D_rf.append(refine_corner)

    return coord_3D_rf, coord_2D_rf


def end_pts_tip2end(x, y, shape):
    m, b = np.polyfit(x, y, 1)
    end_pts = line_end_points_on_image(m, b, shape, True)

    # the distance between 2 possible end pts and index 0 edge

    d1 = np.linalg.norm([x[0] - end_pts[0][0], y[0] - end_pts[0][1]])
    d2 = np.linalg.norm([x[0] - end_pts[1][0], y[0] - end_pts[1][1]])
    if d1 > d2:
        end_pts[0], end_pts[1] = end_pts[1], end_pts[0]
    return end_pts


def cropArea(gray_frame, x, y):
    region_size = 2
    h, w = gray_frame.shape

    tlx = max(round(x - region_size), 0)
    tly = max(round(y - region_size), 0)

    lrx = min(round(x + region_size), w)
    lry = min(round(y + region_size), h)
    region = gray_frame[tly: lry+1, tlx: lrx+1]
    return region

def edge_refinement_conv(gray_frame, x, y, kernel):

    region_size = 6
    h, w = gray_frame.shape

    tlx = max(round(x - region_size), 0)
    tly = max(round(y - region_size), 0)

    lrx = min(round(x + region_size), w)
    lry = min(round(y + region_size), h)
    region = gray_frame[tly: lry, tlx: lrx]


    conv = convolve2D(region, kernel)


    reg_max = np.unravel_index(np.argmax(conv, axis=None), conv.shape)

    # cv2.circle(region, (reg_max[1]+2, reg_max[0]+2), 1, (0, 255, 0), -1)
    # cv2.imshow('123', region)
    # cv2.waitKey(0)
    rf_x = round(x - region_size + reg_max[1] + 2) # 2 is not correct if touching (0, 0)
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


def kernel_choice(m, i, dx, dy):
    s1 = math.tan(math.pi/8)
    s2 = math.tan(math.pi*3/8)
    if -s1 <= m <= s1:
        kernel = np.array([[1, 1, 0, -1, -1],
                           [1, 1, 0, -1, -1],
                           [1, 1, 0, -1, -1],
                           [1, 1, 0, -1, -1],
                           [1, 1, 0, -1, -1]])
        if dx > 0:
            kernel *= -1

    elif s1 < m <= s2:
        kernel = np.array([[-1, -1, -1, -1,  0],
                           [-1, -1, -1,  0,  1],
                           [-1, -1,  0,  1,  1],
                           [-1,  0,  1,  1,  1],
                           [0,   1,  1,  1,  1]])
        if dy < 0:
            kernel *= -1

    elif -s2 <= m < -s1:
        kernel = np.array([[0, -1, -1, -1, -1],
                           [1,  0, -1, -1, -1],
                           [1,  1,  0, -1, -1],
                           [1,  1,  1,  0, -1],
                           [1,  1,  1,  1,  0]])
        if dx > 0:
            kernel *= -1
    else:
        kernel = np.array([[-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]])
        if dy < 0:
            kernel *= -1

    if i % 2 == 0:
        kernel *= -1

    return kernel

if __name__ == "__main__":
    # detect()
    # detect_linear()
    detect_linear_img()