# importing the module
import cv2
import numpy as np
import sys

from needle_utils_temp import camera_para_retrieve, diamond_detection, pose_trans_needle
from Linear_equation_temp import scale_estimation, scale_estimation_4p
# function to display the coordinates of
# of the points clicked on the image

num_pt = 3
coordinates = np.zeros((num_pt, 3), dtype='float64')
index = 0

def click_event(event, x, y, flags, params):
    global coordinates, index
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > button1[0] and y < button1[1] and x > button1[2] and x < button1[3]:
            print('Evaluate!')
            coordinates_ready = coordinates.copy()
            print(coordinates_ready)
            print(trans_tvec[0])

            est_tvec = scale_estimation(coordinates_ready[0], coordinates_ready[1], coordinates_ready[2], 30, 50, mtx)
            print(est_tvec)
            #
            # est_tvec = scale_estimation_4p(coordinates_ready[0], coordinates_ready[1], coordinates_ready[2], coordinates_ready[3], 30, 20, 30, mtx)
            # print(est_tvec)
            trans_error = np.linalg.norm(trans_tvec - est_tvec)
            print(trans_error)

        if y > button2[0] and y < button2[1] and x > button2[2] and x < button2[3]:
            print("current index", index)
            if index >= num_pt - 1:
                index = 0
            else:
                index += 1
            print("new index", index)
        if y > button3[0] and y < button3[1] and x > button3[2] and x < button3[3]:
            coordinates = np.zeros((num_pt, 3), dtype='float64')
            # combine_img = window_init()
            index = 0
            print('Reset')


    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        if index > 3:
            print('All labeled')
        else:
            coordinates[index] = np.array([x, y, 0], dtype='float64')
            print("index {}, x = {}, y = {}".format(index, x, y))
            cv2.putText(combine_img, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(combine_img, (x, y), 2, (0, 0, 255), -1)
            index += 1

        cv2.imshow('image', combine_img)

def window_init():
    img = cv2.imread('../All_images/16.jpg')

    bt_size = 150

    control_image = np.zeros((img.shape[0], bt_size, 3), np.uint8)
    control_image[:bt_size, :bt_size, :] = 180
    cv2.putText(control_image, 'Calculate', (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[bt_size:2 * bt_size, :bt_size, :] = 100
    cv2.putText(control_image, 'Next point', (0, 50 + bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[2 * bt_size:3 * bt_size, :bt_size, :] = 150
    cv2.putText(control_image, 'Clear points', (0, 50 + 2 * bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    combine_img_sub = np.concatenate((img, control_image), axis=1)

    return combine_img_sub


# driver function
if __name__ == "__main__":

    # y 768 x 1024
    img = cv2.imread('../All_images/algorithm_test/3.jpg')

    mtx, dist = camera_para_retrieve()
    diamondCorners, rvec, tvec = diamond_detection(img, mtx, dist)
    if diamondCorners == None:
        print('Try another image')
    trans_tvec = pose_trans_needle(tvec, rvec)  # translation from marker to needle tip

    bt_size = 150
    button1 = [0, bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x
    button2 = [bt_size, 2 * bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x
    button3 = [2 * bt_size, 3 * bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x

    control_image = np.zeros((img.shape[0], bt_size, 3), np.uint8)
    control_image[:bt_size, :bt_size, :] = 180
    cv2.putText(control_image, 'Calculate', (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[bt_size:2 * bt_size, :bt_size, :] = 100
    cv2.putText(control_image, 'Next point', (0, 50 + bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[2 * bt_size:3 * bt_size, :bt_size, :] = 150
    cv2.putText(control_image, 'Clear points', (0, 50 + 2 * bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    combine_img = np.concatenate((img, control_image), axis=1)
    cv2.imshow('image', combine_img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
