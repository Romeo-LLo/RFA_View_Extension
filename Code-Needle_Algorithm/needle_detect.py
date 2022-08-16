import cv2
import numpy as np
import glob
import os
import math
import time
from needle_utils import *
from debug_utils import *
from Linear_equation import scale_estimation
def needle_detect():
    mtx, dist = camera_para_retrieve()

    img_dir = '../All_Images/needle_detect_img'
    imgs_set = glob.glob(os.path.join(img_dir, '*.jpg'))
    for i in range(len(imgs_set)):

        img_path = imgs_set[i]
        print(img_path)
        color_img = cv2.imread(img_path)
        # gray = undistorted(color_img, mtx, dist)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        dilation = edge_suppression(gray)

        diamondCorners, rvec, tvec = diamond_detection(gray, mtx, dist)
        if diamondCorners != None:
            # print('True tvec = ', tvec)
            dilation = generate_mask(diamondCorners, dilation)
            gray = generate_mask(diamondCorners, gray)

        # overlap = cv2.addWeighted(gray, 0.5, dilation, 0.5, 0)
        # cv2.imshow('overlap', overlap)
        # cv2.waitKey(0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines.any() != None:
            pt_set, pixel, deriv = line_differentiator_dispaly(color_img, dilation, lines)
            pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker_display(pt_set, deriv)

            if len(pos_target) > 0 and len(neg_target) > 0:
                refine_pts = line_fit_and_refine_display(pos_target, neg_target, pt_set, gray, color_img)
                # est_tvec = scale_estimation(refine_pts[0], refine_pts[1], refine_pts[2], 10, 10, mtx, dist)
                # print('Est tvec = ', est_tvec)
            # pixel, first_d = line_differentiator_compare(img, dilation, lines)
            # draw_lines(img, edges)
        else:
            print("No lines detected!")

def needle_detect_realtime():
    mtx, dist = camera_para_retrieve()

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    while True:
        ret, frame = cap.read()

        if ret:
            start = time.time()
            color_img = cv2.flip(frame, 0)
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            dilation = edge_suppression(gray)

            diamondCorners, rvec, tvec = diamond_detection(gray, mtx, dist)
            if diamondCorners != None:
                dilation = generate_mask(diamondCorners, dilation)
                gray = generate_mask(diamondCorners, gray)

            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

            if lines is not None:
                pt_set, pixel, deriv = line_differentiator(dilation, lines)
                pos_target, neg_target = edge_checker(pt_set, deriv)

                color_img = rough_edge(pt_set, pos_target, neg_target, color_img)

                # if len(pos_target) > 0 and len(neg_target) > 0:
                #     refine_pts, color_img = line_fit_and_refine(pos_target, neg_target, pt_set, gray, color_img)
                    # print('Found!')
                    # est_tvec = scale_estimation(refine_pts[0], refine_pts[1], refine_pts[2], 10, 10, mtx, dist)
                # else:
                    # print('Not found!')
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            cv2.imshow("Image", color_img)

            end = time.time()
            process_time = end - start
            # print('FPS = ', 1 / xrocess_time)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    # 跳動快的地方，做雜訊綠波
    needle_detect_realtime()

    # needle_detect()

