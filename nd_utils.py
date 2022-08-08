import cv2
import numpy as np
import glob
import os
import math
from test import diamond_detection

from matplotlib import pyplot as plt
from noise_robust_differentiator import derivative_n2
from matplotlib.patches import Circle
from scipy.signal import argrelextrema, find_peaks
import bisect


def camera_para_retrieve():
    mtx = np.load('./CameraParameter/AUX273_mtx.npy')
    dist = np.load('./CameraParameter/AUX273_dist.npy')

    return mtx, dist


def generate_mask(diamondCorners, img):
    sqCorners = diamondCorners[0].squeeze(1)
    sqCorners = sqCorners.astype('int32')
    center = np.mean(sqCorners, axis=0).astype('int32')
    vector = sqCorners - center
    corners = sqCorners + vector * 3

    cv2.fillPoly(img, [corners], (0, 0, 0))
    # cv2.imshow('Mask', img)
    # cv2.waitKey(0)

    return img


def draw_lines(img, mask_img):
    lines = cv2.HoughLines(mask_img, 1, np.pi / 180, 200)
    if lines is not None:
        iter = len(lines)
        for i in range(0, iter):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            length = 1000
            pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
            pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Detected lines", img)
        cv2.waitKey()


def line_average(lines):
    lines = lines.squeeze(1)

    avg_rho, avg_theta = np.mean(lines, axis=0)
    return avg_rho, avg_theta


def line_differentiator(gray_img, color_img, img, lines):
    avg_rho, avg_theta = line_average(lines)

    end_pts = line_end_points_on_image(avg_rho, avg_theta, img.shape, False)
    interval = 200
    x_set = np.linspace(end_pts[0][0], end_pts[1][0], interval)
    y_set = np.linspace(end_pts[0][1], end_pts[1][1], interval)

    pixel = [img[int(y), int(x)] for (y, x) in zip(y_set, x_set)]
    dx = np.sqrt((x_set[1] - x_set[0]) ** 2 + (y_set[1] - y_set[0]) ** 2)
    d_pixel = np.gradient(pixel)
    deriv = d_pixel / dx

    # pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker(, first_d)
    # if len(pos_target) > 0 and len(neg_target) > 0:
    #     line_fit_and_refine(pos_target, neg_target, x_set, y_set, gray_img, color_img)
    # plt.subplot(2, 1, 1)
    # plt.plot(x_set, pixel, color='black', label='True', linestyle='--')
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    # plt.minorticks_on()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x_set, first_d, color='tab:blue', label='grdient')
    # plt.vlines(x_set[pos_target], color='y', ymin=-30, ymax=30)  # vertical
    # plt.vlines(x_set[neg_target], color='purple', ymin=-30, ymax=30)  # vertical
    #
    # plt.vlines(x_set[pos_peaks_arr], color='r', ymin=-20, ymax=20)  # vertical
    # plt.vlines(x_set[neg_peaks_arr], color='g', ymin=-20, ymax=20)  # vertical
    #
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    # plt.minorticks_on()
    # plt.show()

    # img_clone = img.copy()
    # for i, (xx, yy) in enumerate(zip(x_set, y_set)):
    #     cv2.circle(img_clone, (int(xx), int(yy)), 1, (255, 255, 255), -1)
    #     if i % 10 == 0:
    #         cv2.putText(img_clone, str(int(xx)), (int(xx), int(yy)), cv2.FONT_HERSHEY_DUPLEX, 0.6, 50, 1)
    # Hori = np.concatenate((img, img_clone), axis=1)
    # Hori = cv2.resize(Hori, (1600, 600))  # Resize image

    # cv2.imshow('cicle', Hori)
    # cv2.waitKey(0)

    # img_clone = color_img.copy()
    # for index in pos_target:
    #     cv2.circle(img_clone, (int(x_set[index]), int(y_set[index])), 2, (0, 255, 0), -1)
    # for index in neg_target:
    #     cv2.circle(img_clone, (int(x_set[index]), int(y_set[index])), 2, (0, 0, 255), -1)
    #
    # cv2.imshow('cicle', img_clone)
    # cv2.waitKey(0)

    return x_set, y_set, pixel, deriv


def line_differentiator_compare(color_img, img, lines):
    for i in range(lines.shape[0]):
        avg_rho, avg_theta = lines[i][0]
        end_pts = line_end_points_on_image(avg_rho, avg_theta, img.shape)
        interval = 200
        x_set = np.linspace(end_pts[0][0], end_pts[1][0], interval)
        y_set = np.linspace(end_pts[0][1], end_pts[1][1], interval)

        pixel = [img[int(y), int(x)] for (y, x) in zip(y_set, x_set)]
        dx = np.sqrt((x_set[1] - x_set[0]) ** 2 + (y_set[1] - y_set[0]) ** 2)
        d_pixel = np.gradient(pixel)
        first_d = d_pixel / dx

        pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker(x_set, first_d)

        plt.subplot(2, 1, 1)
        plt.plot(x_set, pixel, color='black', label='True', linestyle='--')
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
        plt.minorticks_on()

        plt.subplot(2, 1, 2)
        plt.plot(x_set, first_d, color='tab:blue', label='grdient')
        plt.vlines(x_set[pos_target], color='y', ymin=-30, ymax=30)  # vertical
        plt.vlines(x_set[neg_target], color='purple', ymin=-30, ymax=30)  # vertical

        plt.vlines(x_set[pos_peaks_arr], color='r', ymin=-20, ymax=20)  # vertical
        plt.vlines(x_set[neg_peaks_arr], color='g', ymin=-20, ymax=20)  # vertical

        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
        plt.minorticks_on()
        plt.show()

        img_clone = img.copy()
        for i, (xx, yy) in enumerate(zip(x_set, y_set)):
            cv2.circle(img_clone, (int(xx), int(yy)), 1, (0, 0, 255), -1)

        color_img_clone = color_img.copy()
        for index in pos_target:
            cv2.circle(color_img_clone, (int(x_set[index]), int(y_set[index])), 2, (0, 255, 0), -1)
        for index in neg_target:
            cv2.circle(color_img_clone, (int(x_set[index]), int(y_set[index])), 2, (0, 0, 255), -1)

        Hori = np.concatenate((color_img_clone, cv2.cvtColor(img_clone, cv2.COLOR_GRAY2RGB)), axis=1)
        Hori = cv2.resize(Hori, (1600, 600))  # Resize image

        cv2.imshow('cicle', Hori)
        cv2.waitKey(0)

    return pixel, first_d


def line_fit_and_refine(pos_target, neg_target, x_set, y_set, gray_img, color_img):
    x = np.concatenate((x_set[pos_target], x_set[neg_target]))
    y = np.concatenate((y_set[pos_target], y_set[neg_target]))

    m, b = np.polyfit(x, y, 1)
    end_pts = line_end_points_on_image(m, b, gray_img.shape, True)
    interval = 400
    color_x_set = np.linspace(end_pts[0][0], end_pts[1][0], interval)
    color_y_set = np.linspace(end_pts[0][1], end_pts[1][1], interval)

    pixel = [gray_img[int(y), int(x)] for (y, x) in zip(color_y_set, color_x_set)]
    dx = np.sqrt((color_x_set[1] - color_x_set[0]) ** 2 + (color_y_set[1] - color_y_set[0]) ** 2)
    d_pixel = np.gradient(pixel)
    first_d = d_pixel / dx
    #

    refine_pos_target = inspect_section_peak(pos_target, x_set, color_x_set, first_d, neg=False)
    refine_neg_target = inspect_section_peak(neg_target, x_set, color_x_set, first_d, neg=True)

    plt.subplot(2, 1, 1)
    plt.plot(color_x_set, pixel, color='black', label='True', linestyle='--')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    plt.minorticks_on()

    width = 0.5
    plt.subplot(2, 1, 2)
    plt.plot(color_x_set, first_d, color='tab:blue', label='grdient')
    plt.vlines(x_set[pos_target], color='r', ymin=-30, ymax=30, linewidth=width)  # vertical
    plt.vlines(x_set[neg_target], color='g', ymin=-30, ymax=30, linewidth=width)  # vertical
    plt.vlines(color_x_set[refine_pos_target], color='sandybrown', ymin=-20, ymax=20, linewidth=width)  # vertical
    plt.vlines(color_x_set[refine_neg_target], color='springgreen', ymin=-20, ymax=20, linewidth=width)  # vertical

    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.8)
    plt.minorticks_on()
    plt.show()

    img_clone = color_img.copy()
    # cv2.line(img_clone, end_pts[0], end_pts[1],  (255, 255, 0), 2)
    # cv2.line(img_clone, (int(color_x_set[0]), int(color_y_set[0])), (int(color_x_set[-1]), int(color_y_set[-1])),  (255, 255, 0), 2)

    for (xx, yy) in zip(color_x_set, color_y_set):
        cv2.circle(img_clone, (int(xx), int(yy)), 1, (150, 150, 150), -1)

    for index in pos_target:
        cv2.circle(img_clone, (int(x_set[index]), int(y_set[index])), 1, (0, 0, 255), -1)
    for index in neg_target:
        cv2.circle(img_clone, (int(x_set[index]), int(y_set[index])), 1, (0, 255, 0), -1)

    for index in refine_pos_target:
        cv2.circle(img_clone, (int(color_x_set[index]), int(color_y_set[index])), 1, (80, 127, 255), -1)
    for index in refine_neg_target:
        cv2.circle(img_clone, (int(color_x_set[index]), int(color_y_set[index])), 1, (0, 255, 255), -1)

    cv2.imshow('circle', img_clone)
    cv2.waitKey(0)


def inspect_section_peak(target, x_set, color_x_set, first_d, neg=False):
    interval = len(color_x_set)
    refine_target = []
    window = 8

    if neg:
        first_d = -first_d
    for peak in target:
        closest_x_index = bisect.bisect_left(color_x_set, x_set[peak])

        deriv_inspect = first_d[closest_x_index - window: closest_x_index + window]
        peak_inspect, _ = find_peaks(deriv_inspect, height=(5, 25))
        peak_inspect = list(peak_inspect)
        # filter the deriv that is smaller than the closest x
        closet_deriv = first_d[closest_x_index]
        peak_inspect_greater = [peak for peak in peak_inspect if deriv_inspect[peak] > closet_deriv]

        if len(peak_inspect_greater) == 0:
            refine_target.append(closest_x_index)
        else:
            # find the closet peak according to the center
            refine_peaks_trans = [abs(peak - window) for peak in peak_inspect_greater]
            refine_peak_index = refine_peaks_trans.index(min(refine_peaks_trans))
            index = closest_x_index - window + peak_inspect_greater[refine_peak_index]
            if index > (interval - 1) or index < 0:
                refine_target.append(closest_x_index)
            else:
                refine_target.append(index)
    return refine_target


def edge_checker(x_set, first_d):
    pos_peaks_arr, _ = find_peaks(first_d, height=(5, 12))
    neg_peaks_arr, _ = find_peaks(-first_d, height=(5, 12))
    pos_peaks = list(pos_peaks_arr)
    neg_peaks = list(neg_peaks_arr)

    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        return [], []
    thres = 30
    pixel_thres = 40
    # find the close enough first 2
    pos_target, neg_target, pos_peaks, neg_peaks = init2(x_set, pos_peaks, neg_peaks)
    if len(pos_target) == 0 or len(neg_target) == 0:
        return [], []

    # if (len(pos_target) + len(pos_peaks)) <= 1:
    #     return [], []

    while len(pos_peaks) > 0 and len(neg_peaks) > 0 and len(pos_target) < 3:
        # print('pp', pos_peaks)
        # print('np', neg_peaks)
        # print('pt:', pos_target)
        # print('nt:', neg_target)
        # print('---------------')
        # if (pos_peaks[0] - neg_peaks[0] < pixel_thres) and (neg_peaks[0] - pos_target[-1] < thres):
        if (x_set[pos_peaks[0]] - x_set[neg_peaks[0]] < pixel_thres) and (
                x_set[neg_peaks[0]] - x_set[pos_target[-1]] < pixel_thres):
            pos_target.append(pos_peaks.pop(0))
            neg_target.append(neg_peaks.pop(0))
        else:
            pos_target, neg_target, pos_peaks, neg_peaks = init2(x_set, pos_peaks, neg_peaks)

    # #先用紅的找最近的綠的，下一個綠與紅的檢查距離，過遠的話，以前一個綠的作為第一個重新尋找

    return pos_target, neg_target


def initialization(x_set, pos_peaks, neg_peaks):
    pos_target = []
    neg_target = []
    thres = 75
    if len(pos_peaks) <= 1 or len(neg_peaks) <= 1:
        return [], []

    # check for the first 2 elements, get the first 2 elements
    pos_target.append(pos_peaks.pop(0))
    while neg_peaks[0] < pos_target[0]:
        if len(neg_target) == 0:
            neg_target.append(neg_peaks.pop(0))
        else:
            neg_target[-1] = neg_peaks.pop(0)
        if len(neg_peaks) == 0:
            return [], []
    if len(neg_target) == 0:
        pos_target, neg_target = initialization(x_set, pos_peaks, neg_peaks)

    if x_set[pos_target[0]] - x_set[neg_target[0]] > thres:
        pos_target, neg_target = initialization(x_set, pos_peaks, neg_peaks)

    return pos_target, neg_target


def init2(x_set, pos_peaks, neg_peaks):
    pos_target = []
    neg_target = []
    thres = 75

    if len(pos_peaks) <= 1 or len(neg_peaks) <= 1:
        return [], [], pos_peaks, neg_peaks
    pos_target.append(pos_peaks.pop(0))
    while neg_peaks[0] < pos_target[0]:
        if x_set[pos_target[0]] - x_set[neg_peaks[0]] < thres:
            if len(neg_target) == 0:
                neg_target.append(neg_peaks.pop(0))
            else:
                neg_target[0] = neg_peaks.pop(0)
        else:
            neg_peaks.pop(0)

        if len(neg_peaks) == 0:
            return [], [], pos_peaks, neg_peaks
    if len(neg_target) > 0:
        return pos_target, neg_target, pos_peaks, neg_peaks
    else:
        pos_target, neg_target, pos_peaks, neg_peaks = init2(x_set, pos_peaks, neg_peaks)

    return pos_target, neg_target, pos_peaks, neg_peaks


def edge_supression(img):
    len = 10
    kernel = np.ones((len, len), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)

    # cv2.imshow('img', dilation)
    # cv2.waitKey( 0)

    return dilation


def polar2cartesian(rho: float, theta_rad: float, rotate90: bool = False):
    """
    Converts line equation from polar to cartesian coordinates

    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line

    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b


def line_end_points_on_image(rho: float, theta: float, image_shape: tuple, converted: bool):
    """
    Returns end points of the line on the end of the image
    Args:
        rho: input line rho
        theta: input line theta
        image_shape: shape of the image
        converted: whether input is already m, b
    Returns:
        list: [(x1, y1), (x2, y2)]
    """
    if converted:
        m, b = rho, theta
    else:
        m, b = polar2cartesian(rho, theta, True)

    end_pts = []

    if not np.isclose(m, 0.0):
        x = int(0)
        y = int(solve4y(x, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))
        x = int(image_shape[1] - 1)
        y = int(solve4y(x, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))

    if m is not np.nan:
        y = int(0)
        x = int(solve4x(y, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))
        y = int(image_shape[0] - 1)
        x = int(solve4x(y, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))

    # Sort so that the coordinate x is from smaller to bigger
    sorted_end_pts = sorted(end_pts, key=lambda p: p[0])
    return sorted_end_pts


def solve4x(y: float, m: float, b: float):
    """
    From y = m * x + b
         x = (y - b) / m
    """
    if np.isclose(m, 0.0):
        return 0.0
    if m is np.nan:
        return b
    return (y - b) / m


def solve4y(x: float, m: float, b: float):
    """
    y = m * x + b
    """
    if m is np.nan:
        return b
    return m * x + b


def point_on_image(x: int, y: int, image_shape: tuple):
    """
    Returns true is x and y are on the image
    """
    return 0 <= y < image_shape[0] and 0 <= x < image_shape[1]


def constrast_enhance(image):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1
    beta = 0  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted
