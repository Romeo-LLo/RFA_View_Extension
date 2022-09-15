import cv2
import numpy as np
import glob
import os
import math
import time
from needle_utils import *
from debug_utils import *
from Linear_equation import scale_estimation
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

def needle_detect_raw_realtime():
    mtx, dist = camera_para_retrieve()

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    while True:
        ret, frame = cap.read()

        if ret:
            color_img = cv2.flip(frame, 0)
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            if lines is not None:
                print('lines ', len(lines))
                # for j, line in enumerate(lines):

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break


            cv2.imshow("Image", color_img)

def needle_detect_raw():
    mtx, dist = camera_para_retrieve()

    img_dir = '../All_Images/needle_detect_complex_Img'
    imgs_set = glob.glob(os.path.join(img_dir, '*.bmp'))
    for i in range(len(imgs_set)):

        img_path = imgs_set[i]
        print(img_path)
        color_img = cv2.imread(img_path)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        diamondCorners, rvec, tvec = diamond_detection(gray, mtx, dist)
        if diamondCorners != None:
            gray = generate_mask(diamondCorners, gray)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            print('lines ', len(lines))
            for j, line in enumerate(lines):

                pt_set, pixel, deriv = single_line_differentiator_dispaly(gray, gray, line)
                pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker_display(pt_set, deriv)
                print(len(pos_target))

                # for index in pos_target:
                #     cv2.circle(dilation, (pt_set[index][0], pt_set[index][1]), 2, (0, 0, 255), -1)
                # for index in neg_target:
                #     cv2.circle(dilation, (pt_set[index][0], pt_set[index][1]), 2, (0, 255, 0), -1)
            # cv2.imshow('cicle', dilation)
            # cv2.waitKey(0)

            # if len(pos_target) < 2:
            #     continue
            # else:
            #     color_img = rough_edge_display(pt_set, pos_target, neg_target, color_img, j)
            # cv2.imshow('circle', color_img)
            # cv2.waitKey(0)


        else:
            print("No lines detected!")
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
def needle_detect_each_line():
    mtx, dist = camera_para_retrieve()
    img_dir = '../All_Images/needle_detect_white_Img'
    imgs_set = glob.glob(os.path.join(img_dir, '*.jpg'))
    for i in range(len(imgs_set)):

        img_path = imgs_set[i]
        color_img = cv2.imread(img_path)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        diamondCorners, rvec, tvec = diamond_detection(gray, mtx, dist)
        aruco_detect = False
        if diamondCorners != None:
            aruco_detect = True
            gray = generate_mask(diamondCorners, gray)


        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        pos_pts_all_lines = np.empty((0, 3, 2), dtype=int)
        neg_pts_all_lines = np.empty((0, 3, 2), dtype=int)
        center_coord_cluster = np.empty((0, 2), dtype=int)
        do_cluster = False

        if lines is not None:
            for j, line in enumerate(lines):
                for k in range(3):
                    kernel_len = k * 4 + 8
                    dilation = edge_suppression(gray, kernel_len)
                    cv2.imshow('circle', dilation)
                    cv2.waitKey(0)

                    pt_set, pixel, deriv = single_line_differentiator_dispaly(dilation, dilation, line)
                    pos_target, neg_target, pos_paks_arr, neg_peaks_arr = edge_checker_display(pt_set, deriv)
                    if len(pos_target) <= 2:
                        continue
                    else:
                        do_cluster = True
                        pos_pts_one_line = coordinate_generator(pt_set, pos_target)
                        neg_pts_one_line = coordinate_generator(pt_set, neg_target)
                        center_pt = center_pt_generator(pos_pts_one_line, neg_pts_one_line)

                        pos_pts_all_lines = np.concatenate([pos_pts_all_lines, pos_pts_one_line])
                        neg_pts_all_lines = np.concatenate([neg_pts_all_lines, neg_pts_one_line])
                        center_coord_cluster = np.concatenate([center_coord_cluster, center_pt])


            if do_cluster:
                # to remove points that are not at same location
                target_index = center_pt_cluster(center_coord_cluster)
                pos_coord_cluster = pos_pts_all_lines[target_index, :, :]
                pos_coord_mean = np.mean(pos_coord_cluster, axis=1).squeeze(axis=0)
                neg_coord_cluster = neg_pts_all_lines[target_index, :, :]
                neg_coord_mean = np.mean(neg_coord_cluster, axis=1).squeeze(axis=0)

                pos_coord_mean_3D = []
                neg_coord_mean_3D = []
                for i in range(3):
                    pt_pos = np.array([pos_coord_mean[i][0], pos_coord_mean[i][1], 0], dtype='float64')
                    pt_neg = np.array([neg_coord_mean[i][0], neg_coord_mean[i][1], 0], dtype='float64')

                    pos_coord_mean_3D.append(pt_pos)
                    neg_coord_mean_3D.append(pt_neg)
        #
        #         est_tvec = scale_estimation(pos_coord_mean_3D[0], pos_coord_mean_3D[1], pos_coord_mean_3D[2], 20, 20, mtx, dist)
        #         if aruco_detect:
        #             print(tvec)

                for coord in pos_coord_mean:
                    color_img = cv2.circle(color_img, (int(coord[0]), int(coord[1])), 2, [0, 255, 0], -1)
                for coord in neg_coord_mean:
                    color_img = cv2.circle(color_img, (int(coord[0]), int(coord[1])), 2, [0, 0, 255], -1)
                cv2.imshow('circle', color_img)
                cv2.waitKey(0)
        # else:
        #     print("No lines detected!")
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
            dilation = edge_suppression(gray, 10)

            diamondCorners, rvec, tvec = diamond_detection(gray, mtx, dist)
            if diamondCorners != None:
                print(tvec)
            #     aruco_detect = True
            #     dilation = generate_mask(diamondCorners, dilation)
            #     gray = generate_mask(diamondCorners, gray)
            #
            # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            #
            # pos_pts_all_lines = np.empty((0, 3, 2), dtype=int)
            # neg_pts_all_lines = np.empty((0, 3, 2), dtype=int)
            # center_coord_cluster = np.empty((0, 2), dtype=int)
            # do_cluster = False
            # if lines is not None:
            #     for j, line in enumerate(lines):
            #
            #         pt_set, pixel, deriv = single_line_differentiator_dispaly(dilation, dilation, line)
            #         pos_target, neg_target, pos_peaks_arr, neg_peaks_arr = edge_checker_display(pt_set, deriv)
            #         print(len(pos_peaks_arr))
            #         if len(pos_target) <= 2:
            #             continue
            #         else:
            #             do_cluster = True
            #             pos_pts_one_line = coordinate_generator(pt_set, pos_target)
            #             neg_pts_one_line = coordinate_generator(pt_set, neg_target)
            #             center_pt = center_pt_generator(pos_pts_one_line, neg_pts_one_line)
            #
            #             pos_pts_all_lines = np.concatenate([pos_pts_all_lines, pos_pts_one_line])
            #             neg_pts_all_lines = np.concatenate([neg_pts_all_lines, neg_pts_one_line])
            #             center_coord_cluster = np.concatenate([center_coord_cluster, center_pt])
            #
            #         if do_cluster:
            #             # to remove points that are not at same location
            #             target_index = center_pt_cluster(center_coord_cluster)
            #             pos_coord_cluster = pos_pts_all_lines[target_index, :, :]
            #             pos_coord_mean = np.mean(pos_coord_cluster, axis=1).squeeze(axis=0)
            #             neg_coord_cluster = neg_pts_all_lines[target_index, :, :]
            #             neg_coord_mean = np.mean(neg_coord_cluster, axis=1).squeeze(axis=0)
            #
            #             for coord in pos_coord_mean:
            #                 color_img = cv2.circle(color_img, (coord[0], coord[1]), 2, [0, 255, 0], -1)
            #             for coord in neg_coord_mean:
            #                 color_img = cv2.circle(color_img, (coord[0], coord[1]), 2, [0, 0, 255], -1)
            #
            #             pos_coord_mean_3D = []
            #             neg_coord_mean_3D = []
            #             for i in range(3):
            #                 pt_pos = np.array([pos_coord_mean[i][0], pos_coord_mean[i][1], 0], dtype='float64')
            #                 pt_neg = np.array([neg_coord_mean[i][0], neg_coord_mean[i][1], 0], dtype='float64')
            #
            #                 pos_coord_mean_3D.append(pt_pos)
            #                 neg_coord_mean_3D.append(pt_neg)
            #
            #             est_tvec = scale_estimation(pos_coord_mean_3D[0], pos_coord_mean_3D[1], pos_coord_mean_3D[2], 20, 20,
            #                                         mtx, dist)
            #             if aruco_detect:
            #                 print(tvec)


            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            cv2.imshow("Image", gray)

            end = time.time()
            process_time = end - start
            # print('FPS = ', 1 / xrocess_time)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    # 跳動快的地方，做雜訊綠波
    needle_detect_realtime()
    # needle_detect_each_line()
    # needle_detect()
    # needle_detect_raw()

