
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
from edge_refinement import *
import torch
import gi
import cv2.aruco as aruco
import datetime

gi.require_version("Gst", "1.0")
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_final4.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)


plist = [1, 4, 8]
dlist = [48.262, 57.915]
tip_off = 2.25

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
aruco_error_record

def needle_refinement_error_record():
    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    e_list = [0] * 2
    angle_list = [0] * 2

    eListcur = [0] * 2

    anchor = 0
    count = 0
    angle_count = 0


    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):
                    m, b = line_polyfit(x, y)
                    p1 = (0, round(b))
                    p2 = (1000, round(m * 1000 + b))
                    cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

                    coord_3D = []
                    coord_3D_rf, coord_2D_rf = edge_refinement_linear(gray_frame, x, y, plist)

                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                    tip, end = scale_estimation_multi_mod(coord_3D[0], coord_3D[1], coord_3D[2], dlist[0], dlist[1],
                                                      mtx, tip_off)
                    tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0],
                                                      dlist[1], mtx, tip_off)
                    error = error_calc_board(tip, anchor=anchor)
                    error_rf = error_calc_board(tip_rf, anchor=anchor)

                    e_list[0] += error
                    e_list[1] += error_rf

                    eListcur[0] = round(error, 2)
                    eListcur[1] = round(error_rf, 2)
                    count += 1
                    eList = [round(x / count, 2) for x in e_list]
                    print(eListcur, eList, count)



                    diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)
                    if diamondCorners:
                        tip_t, end_t = pose_trans_needle(tvec, rvec)
                        angle_error, dist_error = error_calc(tip_t, end_t, tip, end)
                        angle_error_rf, dist_error_rf = error_calc(tip_t, end_t, tip_rf, end_rf)

                        angle_list[0] += angle_error
                        angle_list[1] += angle_error_rf
                        angle_count += 1
                        angleList = [round(x / angle_count, 2) for x in angle_list]
                        print(angleList, angle_count)




        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            anchor += 1
            print(f"Now point to hole {anchor}")


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # aruco_error_record()
    needle_refinement_error_record()