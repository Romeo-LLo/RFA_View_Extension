import numpy as np
import os, json, cv2, random
import TIS
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import gi
import cv2.aruco as aruco
import datetime
from needle_utils import *
from Linear_equation import *
from edge_refinement import *


def manual_record():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    mtx, dist = camera_para_retrieve()
    est_state = False

    while True:
        if Tis.Snap_image(1) is True:

            rawframe = Tis.Get_image()
            rawframe = rawframe[:, :, :3]
            dis_frame = np.array(rawframe)
            frame = undistort_img(dis_frame, mtx, dist)


            # diamondCorners, rvec, tvec = diamond_detection(frame, mtx, dist)

            if est_state:

                outputPath = '../All_images/TestImg1222'
                ts = datetime.datetime.now()
                filename = "{}.jpg".format(ts.strftime("%M-%S"))
                path = os.path.sep.join((outputPath, filename))
                cv2.imwrite(path, frame)
                print('record')
                est_state = False

        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            est_state = True
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def automatic_record():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    count = 0
    est_state = False
    while True:
        if Tis.Snap_image(1) is True:

            rawframe = Tis.Get_image()
            rawframe = rawframe[:, :, :3]
            dis_frame = np.array(rawframe)
            if count < 500 and est_state:
                outputPath = '../All_images/record/hole4'
                ts = datetime.datetime.now()
                filename = "{}-{}.jpg".format(ts.strftime("%M-%S"), count)
                path = os.path.sep.join((outputPath, filename))
                cv2.imwrite(path, dis_frame)
                count += 1
                print(count)

            frameS = cv2.resize(dis_frame, (1080, 810))
            cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            est_state = True
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # automatic_record()
    manual_record()