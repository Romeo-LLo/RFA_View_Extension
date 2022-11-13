
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



def aruco_error_record():

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    avg_error = 0
    frame_count = 0
    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)


            if diamondCorners:
                tip_t, end_t = pose_trans_needle(tvec, rvec)
                error = error_calc_board(tip_t, 0)
                avg_error += error
                frame_count += 1
                print(frame_count, avg_error)


        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)
        if cv2.waitKey(1) == ord('q'):
            print(frame_count)
            print(avg_error / frame_count)

            break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    aruco_error_record()
