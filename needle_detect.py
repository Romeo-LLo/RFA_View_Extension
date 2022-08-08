import cv2
import numpy as np
import glob
import os
import math
from test import diamond_detection

from matplotlib import pyplot as plt

from noise_robust_differentiator import derivative_n2
from nd_utils import *


def needle_detect():
    mtx, dist = camera_para_retrieve()

    img_dir = './needle_detect_img'
    imgs_set = glob.glob(os.path.join(img_dir, '*.jpg'))
    for i in range(len(imgs_set)):
    # for i in range(11, 12):

        # img_path = './needle_detect_img/2022-07-18_11-35-12.jpg'
        img_path = imgs_set[i]
        print(img_path)
        color_img = cv2.imread(img_path)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        raw_gray = gray.copy()
        dilation = edge_supression(gray)

        diamondCorners, rvec, tvec = diamond_detection(gray, mtx, dist)
        if diamondCorners != None:
            dilation = generate_mask(diamondCorners, dilation)
            gray = generate_mask(diamondCorners, gray)

        # overlap = cv2.addWeighted(gray, 0.5, dilation, 0.5, 0)
        # cv2.imshow('overlap', overlap)
        # cv2.waitKey(0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines.any() != None:
            x_set, y_set, pixel, deriv = line_differentiator(raw_gray, color_img, dilation, lines)
            pos_target, neg_target = edge_checker(x_set, deriv)


            if len(pos_target) > 0 and len(neg_target) > 0:
                line_fit_and_refine(pos_target, neg_target, x_set, y_set, raw_gray, color_img)

            # pixel, first_d = line_differentiator_compare(img, dilation, lines)
            # draw_lines(img, edges)
        else:
            print("No lines detected!")


if __name__ == "__main__":

    # 跳動快的地方，做雜訊綠波
    needle_detect()
