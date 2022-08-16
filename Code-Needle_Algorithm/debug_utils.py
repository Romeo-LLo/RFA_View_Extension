import cv2
import numpy as np
import glob
import os
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import argrelextrema, find_peaks
import bisect
from bresenham import bresenham
from needle_utils import *

pixel_thres = 50
peak_lower_bound = 20
peak_upper_bound = 60
window = 8
line_height = 50
line_height_target = 80

def line_differentiator_dispaly(color_img, dilation, lines):
    avg_rho, avg_theta = line_average(lines)

    end_pts = line_end_points_on_image(avg_rho, avg_theta, dilation.shape, False)
    pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    pt_set = np.array(pt_set)


    pixel = [dilation[pt[1], pt[0]] for pt in pt_set]
    d_pixel = np.gradient(pixel)
    pt_index = np.linspace(1, len(pixel), len(pixel))

    pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker_display(pt_set, d_pixel)
    # if len(pos_target) > 0 and len(neg_target) > 0:
    #     line_fit_and_refine(pos_target, neg_target, x_set, y_set, gray_img, color_img)

    plt.subplot(2, 1, 1)
    plt.plot(pt_index, pixel, color='black', label='True', linestyle='--')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    plt.minorticks_on()

    #


    plt.subplot(2, 1, 2)
    plt.plot(pt_index, d_pixel, color='tab:blue', label='grdient')
    plt.vlines(pt_index[pos_target], color='y', ymin=-line_height_target, ymax=line_height_target)  # vertical
    plt.vlines(pt_index[neg_target], color='purple', ymin=-line_height_target, ymax=line_height_target)  # vertical

    plt.vlines(pt_index[pos_peaks_arr], color='r', ymin=-line_height, ymax=line_height)  # vertical
    plt.vlines(pt_index[neg_peaks_arr], color='g', ymin=-line_height, ymax=line_height)  # vertical

    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    plt.minorticks_on()
    plt.show()


    # img_clone = color_img.copy()
    # for index in pos_peaks_arr:
    #     cv2.circle(img_clone, (pt_set[index][0], pt_set[index][1]), 2, (0, 0, 255), -1)
    # for index in neg_peaks_arr:
    #     cv2.circle(img_clone, (pt_set[index][0], pt_set[index][1]), 2, (0, 255, 0), -1)
    #
    # cv2.imshow('cicle', img_clone)
    # cv2.waitKey(0)

    return pt_set, pixel, d_pixel



def edge_checker_display(pt_set, first_d):

    pos_peaks_arr, _ = find_peaks(first_d, height=(peak_lower_bound, peak_upper_bound))
    neg_peaks_arr, _ = find_peaks(-first_d, height=(peak_lower_bound, peak_upper_bound))
    pos_peaks = list(pos_peaks_arr)
    neg_peaks = list(neg_peaks_arr)

    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        return [], [], pos_peaks_arr, neg_peaks_arr
    pos_target, neg_target, pos_peaks, neg_peaks = init(pt_set, pos_peaks, neg_peaks)
    if len(pos_target) == 0 or len(neg_target) == 0:
        return [], [], pos_peaks_arr, neg_peaks_arr

    # if (len(pos_target) + len(pos_peaks)) <= 1:
    #     return [], []

    while len(pos_peaks) > 0 and len(neg_peaks) > 0 and len(pos_target) < 3 and len(pos_target) > 0:
        pending_dist = pixel_distance(pt_set, pos_peaks[0], neg_peaks[0])
        adding_dist = pixel_distance(pt_set, neg_peaks[0], pos_target[-1])

        if pending_dist < pixel_thres and adding_dist < pixel_thres:
            pos_target.append(pos_peaks.pop(0))
            neg_target.append(neg_peaks.pop(0))
        else:
            pos_target, neg_target, pos_peaks, neg_peaks = init(pt_set, pos_peaks, neg_peaks)

    return pos_target, neg_target, pos_peaks_arr, neg_peaks_arr





def line_fit_and_refine_display(pos_target, neg_target, pt_set, gray_img, color_img):
    rough_peaks = np.concatenate((pt_set[pos_target], pt_set[neg_target]))
    x, y = rough_peaks.T
    m, b = np.polyfit(x, y, 1)
    end_pts = line_end_points_on_image(m, b, gray_img.shape, True)

    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    pixel = [gray_img[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = np.gradient(pixel)

    gray_pt_set = np.array(gray_pt_set)


    refine_pos_target, closest_pos_record = inspect_section_peak_display(pos_target, pt_set, gray_pt_set, d_pixel, neg=False)
    refine_neg_target, closest_neg_record = inspect_section_peak_display(neg_target, pt_set, gray_pt_set, d_pixel, neg=True)
    pt_index = np.linspace(1, len(pixel), len(pixel))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(pt_index, pixel, color='black', label='True', linestyle='--')
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    # plt.minorticks_on()

    width = 0.5
    # plt.subplot(2, 1, 2)
    plt.plot(pt_index, d_pixel, color='tab:blue', label='grdient')
    plt.vlines(gray_pt_set[closest_pos_record], color='r', ymin=-line_height, ymax=line_height, linewidth=width)  # vertical
    plt.vlines(gray_pt_set[closest_neg_record], color='g', ymin=-line_height, ymax=line_height, linewidth=width)  # vertical
    plt.vlines(gray_pt_set[refine_pos_target], color='sandybrown', ymin=-line_height_target, ymax=line_height_target, linewidth=width)  # vertical
    plt.vlines(gray_pt_set[refine_neg_target], color='springgreen', ymin=-line_height_target, ymax=line_height_target, linewidth=width)  # vertical

    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    plt.minorticks_on()
    plt.show()

    img_clone = color_img.copy()

    # for coord in color_pt_set:
    #     cv2.circle(img_clone, (coord[0], coord[1]), 1, (150, 150, 150), -1)

    for index in pos_target:
        cv2.circle(img_clone, (pt_set[index][0], pt_set[index][1]), 1, (0, 0, 255), -1)
    for index in neg_target:
        cv2.circle(img_clone, (pt_set[index][0], pt_set[index][1]), 1, (0, 255, 0), -1)

    for index in refine_pos_target:
        cv2.circle(img_clone, (gray_pt_set[index][0], gray_pt_set[index][1]), 1, (80, 127, 255), -1)
    for index in refine_neg_target:
        cv2.circle(img_clone, (gray_pt_set[index][0], gray_pt_set[index][1]), 1, (0, 255, 255), -1)

    cv2.imshow('circle', img_clone)
    cv2.waitKey(0)


def inspect_section_peak_display(target, pt_set, gray_pt_set, d_pixel, neg=False):
    interval = len(gray_pt_set)
    refine_target = []
    closest_record = []
    if neg:
        d_pixel = -d_pixel
    for peak in target:
        closest_index = find_closest_index(gray_pt_set, pt_set, peak)
        closest_record.append(closest_index)
        deriv_inspect = d_pixel[closest_index - window: closest_index + window]
        peak_inspect, _ = find_peaks(deriv_inspect, height=(peak_lower_bound, peak_upper_bound))
        peak_inspect = list(peak_inspect)
        # filter the deriv that is smaller than the closest x
        closet_deriv = d_pixel[closest_index]
        peak_inspect_greater = [peak for peak in peak_inspect if deriv_inspect[peak] > closet_deriv]

        if len(peak_inspect_greater) == 0:
            refine_target.append(closest_index)
        else:
            # find the closet peak according to the center
            refine_peaks_trans = [abs(peak - window) for peak in peak_inspect_greater]
            refine_peak_index = refine_peaks_trans.index(min(refine_peaks_trans))
            index = closest_index - window + peak_inspect_greater[refine_peak_index]
            if index > (interval - 1) or index < 0:
                refine_target.append(closest_index)
            else:
                refine_target.append(index)
    return refine_target, closest_record




# def line_differentiator_compare(color_img, img, lines):
#     for i in range(lines.shape[0]):
#         avg_rho, avg_theta = lines[i][0]
#         end_pts = line_end_points_on_image(avg_rho, avg_theta, img.shape)
#         interval = 200
#         x_set = np.linspace(end_pts[0][0], end_pts[1][0], interval)
#         y_set = np.linspace(end_pts[0][1], end_pts[1][1], interval)
#
#         pixel = [img[int(y), int(x)] for (y, x) in zip(y_set, x_set)]
#         dx = np.sqrt((x_set[1] - x_set[0]) ** 2 + (y_set[1] - y_set[0]) ** 2)
#         d_pixel = np.gradient(pixel)
#         first_d = d_pixel / dx
#
#         pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker(x_set, first_d)
#
#         plt.subplot(2, 1, 1)
#         plt.plot(x_set, pixel, color='black', label='True', linestyle='--')
#         plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
#         plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
#         plt.minorticks_on()
#
#         plt.subplot(2, 1, 2)
#         plt.plot(x_set, first_d, color='tab:blue', label='grdient')
#         plt.vlines(x_set[pos_target], color='y', ymin=-30, ymax=30)  # vertical
#         plt.vlines(x_set[neg_target], color='purple', ymin=-30, ymax=30)  # vertical
#
#         plt.vlines(x_set[pos_peaks_arr], color='r', ymin=-20, ymax=20)  # vertical
#         plt.vlines(x_set[neg_peaks_arr], color='g', ymin=-20, ymax=20)  # vertical
#
#         plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
#         plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
#         plt.minorticks_on()
#         plt.show()
#
#         img_clone = img.copy()
#         for i, (xx, yy) in enumerate(zip(x_set, y_set)):
#             cv2.circle(img_clone, (int(xx), int(yy)), 1, (0, 0, 255), -1)
#
#         color_img_clone = color_img.copy()
#         for index in pos_target:
#             cv2.circle(color_img_clone, (int(x_set[index]), int(y_set[index])), 2, (0, 255, 0), -1)
#         for index in neg_target:
#             cv2.circle(color_img_clone, (int(x_set[index]), int(y_set[index])), 2, (0, 0, 255), -1)
#
#         Hori = np.concatenate((color_img_clone, cv2.cvtColor(img_clone, cv2.COLOR_GRAY2RGB)), axis=1)
#         Hori = cv2.resize(Hori, (1600, 600))  # Resize image
#
#         cv2.imshow('cicle', Hori)
#         cv2.waitKey(0)
#
#     return pixel, first_d