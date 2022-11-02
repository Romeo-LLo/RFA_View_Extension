# importing the module
import cv2
import numpy as np
import sys

from needle_utils_temp import *
from Linear_equation_temp import scale_estimation, scale_estimation_4p
# function to display the coordinates of
# of the points clicked on the image

num_pt = 3
# coordinates = np.zeros((num_pt, 3), dtype='float64')
coordinates = np.array([[628, 838, 0],
                        [535, 701, 0],
                        [443, 569, 0]], dtype='float64')
index = 0

def click_event(event, x, y, flags, params):
    global coordinates, index
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > button1[0] and y < button1[1] and x > button1[2] and x < button1[3]:
            print('Evaluate!')
            coordinates_ready = coordinates.copy()
            print(coordinates_ready)
            tip_t = pose_trans_needle(tvec, rvec, 21.2)
            end_t = pose_trans_needle(tvec, rvec, 3)
            tip, end = scale_estimation(coordinates_ready[0], coordinates_ready[1], coordinates_ready[2], 40, 40, mtx)

            error_calc(tip_t, end_t, tip, end)

        if y > button2[0] and y < button2[1] and x > button2[2] and x < button2[3]:
            print("current index", index)
            if index >= num_pt - 1:
                index = 0
            else:
                index += 1
            print("new index", index)
        if y > button3[0] and y < button3[1] and x > button3[2] and x < button3[3]:
            coordinates = np.zeros((num_pt, 3), dtype='float64')
            index = 0
            print('Reset')

        if y > button4[0] and y < button4[1] and x > button4[2] and x < button4[3]:

            tip_t = pose_trans_needle(tvec, rvec, 21.2)
            end_t = pose_trans_needle(tvec, rvec, 3)

            coordinates_ready = coordinates.copy()
            coordinates_copy = coordinates.copy()

            print(coordinates_ready)
            tip, end = scale_estimation(coordinates_ready[0], coordinates_ready[1], coordinates_ready[2], 40, 40, mtx)


            coordinates_fit = line_fit(coordinates_copy)
            print(coordinates_fit)

            tip_f, end_f = scale_estimation(coordinates_fit[0], coordinates_fit[1], coordinates_fit[2], 40, 40, mtx)
            for i in range(num_pt):
                x = int(coordinates_fit[i][0])
                y = int(coordinates_fit[i][1])
                cv2.putText(combine_img, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(combine_img, (x, y), 2, (255, 0, 0), -1)

            print('Compare')
            print('Without fit')
            error_calc(tip_t, end_t, tip, end)
            print('With fit')
            error_calc(tip_f, end_f, tip, end)



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
    img = cv2.imread('../All_images/frame0.jpg')

    mtx, dist = camera_para_retrieve()
    diamondCorners, rvec, tvec = diamond_detection(img, mtx, dist)
    if diamondCorners == None:
        print('Try another image')

    trans_tvec = pose_trans_needle(tvec, rvec)  # translation from marker to needle tip

    bt_size = 150
    button1 = [0, bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x
    button2 = [bt_size, 2 * bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x
    button3 = [2 * bt_size, 3 * bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x
    button4 = [3 * bt_size, 4 * bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x

    control_image = np.zeros((img.shape[0], bt_size, 3), np.uint8)
    control_image[:bt_size, :bt_size, :] = 180
    cv2.putText(control_image, 'Calculate', (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[bt_size:2 * bt_size, :bt_size, :] = 100
    cv2.putText(control_image, 'Next point', (0, 50 + bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[2 * bt_size:3 * bt_size, :bt_size, :] = 150
    cv2.putText(control_image, 'Clear points', (0, 50 + 2 * bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    control_image[3 * bt_size:4 * bt_size, :bt_size, :] = 100
    cv2.putText(control_image, 'Compare fit', (0, 50 + 3 * bt_size), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    combine_img = np.concatenate((img, control_image), axis=1)
    cv2.imshow('image', combine_img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
