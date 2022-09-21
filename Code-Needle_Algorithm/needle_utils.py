import cv2
import numpy as np
import glob
import os
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import argrelextrema, find_peaks
import bisect
import cv2.aruco as aruco
# from bresenham import bresenham

pixel_lower_thres = 20
pixel_upper_thres = 50
peak_lower_bound = 10
peak_upper_bound = 60
window = 8
line_height = 50
line_height_target = 80

def camera_para_retrieve():
    mtx = np.load('../CameraParameter/AUX273_mtx2.npy')
    dist = np.load('../CameraParameter/AUX273_dist2.npy')

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


def line_differentiator(dilation, lines):
    avg_rho, avg_theta = line_average(lines)

    end_pts = line_end_points_on_image(avg_rho, avg_theta, dilation.shape, False)
    # there is still a big here, end_pts sometimes is empty
    pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    pt_set = np.array(pt_set)
    pixel = [dilation[pt[1], pt[0]] for pt in pt_set]

    d_pixel = np.gradient(pixel)

    # return d_pixel is maybe better, need distance invariant
    return pt_set, pixel, d_pixel


def edge_checker(pt_set, first_d):


    pos_peaks_arr, _ = find_peaks(first_d, height=(peak_lower_bound, peak_upper_bound))
    neg_peaks_arr, _ = find_peaks(-first_d, height=(peak_lower_bound, peak_upper_bound))
    pos_peaks = list(pos_peaks_arr)
    neg_peaks = list(neg_peaks_arr)

    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        return [], []

    pos_target, neg_target, pos_peaks, neg_peaks = init(pt_set, pos_peaks, neg_peaks)
    if len(pos_target) == 0 or len(neg_target) == 0:
        return [], []

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

    return pos_target, neg_target


def pixel_distance(pt_set, index1, index2):
    pixel_distance = np.sqrt((pt_set[index1][0] - pt_set[index2][0])**2 + (pt_set[index1][1] - pt_set[index2][1])**2)
    return pixel_distance


def init(pt_set, pos_peaks, neg_peaks):
    pos_target = []
    neg_target = []

    if len(pos_peaks) <= 1 or len(neg_peaks) <= 1:
        return [], [], pos_peaks, neg_peaks
    pos_target.append(pos_peaks.pop(0))
    while neg_peaks[0] < pos_target[0]:
        if pixel_distance(pt_set, pos_target[0], neg_peaks[0]) < pixel_upper_thres and pixel_distance(pt_set, pos_target[0], neg_peaks[0]) > pixel_lower_thres:
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
        pos_target, neg_target, pos_peaks, neg_peaks = init(pt_set, pos_peaks, neg_peaks)

    return pos_target, neg_target, pos_peaks, neg_peaks



def line_fit_and_refine(pos_target, neg_target, pt_set, gray_img, color_img):
    rough_peaks = np.concatenate((pt_set[pos_target], pt_set[neg_target]))
    x, y = rough_peaks.T
    m, b = np.polyfit(x, y, 1)
    end_pts = line_end_points_on_image(m, b, gray_img.shape, True)

    gray_pt_set = list(bresenham(end_pts[0][0], end_pts[0][1], end_pts[1][0], end_pts[1][1]))
    pixel = [gray_img[pt[1], pt[0]] for pt in gray_pt_set]
    d_pixel = np.gradient(pixel)
    gray_pt_set = np.array(gray_pt_set)

    refine_pos_target = inspect_section_peak(pos_target, pt_set, gray_pt_set, d_pixel, neg=False)
    refine_neg_target = inspect_section_peak(neg_target, pt_set, gray_pt_set, d_pixel, neg=True)
    refine_pts = []
    for i in range(len(refine_pos_target)):
        pt_neg = np.array([gray_pt_set[refine_neg_target[i]][0], gray_pt_set[refine_neg_target[i]][1], 0], dtype='float64')
        pt_pos = np.array([gray_pt_set[refine_pos_target[i]][0], gray_pt_set[refine_pos_target[i]][1], 0], dtype='float64')
        refine_pts.append(pt_neg)
        refine_pts.append(pt_pos)


    for index in refine_pos_target:
        color_img = cv2.circle(color_img, (gray_pt_set[index][0], gray_pt_set[index][1]), 5, (0, 0, 255), -1)
    for index in refine_neg_target:
        color_img = cv2.circle(color_img, (gray_pt_set[index][0], gray_pt_set[index][1]), 5, (0, 255, 0), -1)

    return refine_pts, color_img


def inspect_section_peak(target, pt_set, gray_pt_set, d_pixel, neg=False):
    interval = len(gray_pt_set)
    refine_target = []
    if neg:
        d_pixel = -d_pixel
    for peak in target:
        closest_index = find_closest_index(gray_pt_set, pt_set, peak)
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
    return refine_target


def find_closest_index(gray_pt_set, pt_set, peak):
    rough_coord = pt_set[peak]
    dist = np.sum((gray_pt_set - rough_coord)**2, axis=1)
    return np.argmin(dist)

def rough_edge(pt_set, pos_target, neg_target, color_img):

    for index in pos_target:
        color_img = cv2.circle(color_img, (pt_set[index][0], pt_set[index][1]), 5, (0, 0, 255), -1)
    for index in neg_target:
        color_img = cv2.circle(color_img, (pt_set[index][0], pt_set[index][1]), 5, (0, 255, 0), -1)

    return color_img


def edge_suppression(img, kernel_len):


    kernel = np.ones((kernel_len, kernel_len), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)

    # cv2.imshow('img', dilation)
    # cv2.waitKey(0)

    return dilation

def edge_suppression_erosion(img):

    len = 15
    kernel = np.ones((len, len), np.uint8)
    img1 = cv2.dilate(img, kernel, iterations=1)
    img2 = cv2.erode(img, kernel, iterations=1)


    cv2.imshow('img', img1)
    cv2.waitKey(0)

    return img



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
            if (x, y) not in end_pts:
                end_pts.append((x, y))

        x = int(image_shape[1] - 1)
        y = int(solve4y(x, m, b))
        if point_on_image(x, y, image_shape):
            if (x, y) not in end_pts:
                end_pts.append((x, y))

    if m is not np.nan:
        y = int(0)
        x = int(solve4x(y, m, b))
        if point_on_image(x, y, image_shape):
            if (x, y) not in end_pts:
                end_pts.append((x, y))

        y = int(image_shape[0] - 1)
        x = int(solve4x(y, m, b))
        if point_on_image(x, y, image_shape):
            if (x, y) not in end_pts:
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


def contrast_enhance(image):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1
    beta = 0  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted

# def undistort(img, mtx, dist):
#
#     h, w = img.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#     undist = cv2.undistort(img, mtx, dist, None, newcameramtx)
#     return undist

def undistort_img(img, mtx, dist):

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def diamond_detection(img, mtx, dist):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.3
    markerLength = 0.9
    arucoParams = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

    if np.any(ids != None):
        diamondCorners, diamondIds = aruco.detectCharucoDiamond(img, corners, ids,
                                                                squareLength / markerLength)

        if np.any(diamondIds != None):  # if aruco marker detected
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, mtx, dist)  # For a single marker
            # rvec, tvec = solvePnP()
            return diamondCorners, rvec, tvec
        else:
            return None, None, None
    else:
        return None, None, None

def pose_trans_needle(tvec, rvec):
    r_matrix, _ = cv2.Rodrigues(rvec[0][0])
    trans = np.matmul(r_matrix, np.array([[0], [21], [2.5]]))
    needle_tvec = tvec[0][0] + trans.T
    return needle_tvec


def corner_refinement(src_gray, corners):
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    # Calculate the refined corner locations
    rf_corners = cv2.cornerSubPix(src_gray, corners, winSize, zeroZone, criteria)

    return rf_corners