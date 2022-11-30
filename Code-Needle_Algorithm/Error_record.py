
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

lazer = 1.036
# plist = [1, 4, 8]
# dlist = [48.262, 57.915]
plist = [1, 3, 6]
dlist = [40, 50]
dlist = [d / lazer for d in dlist]
tip_off = 2.25

def aruco_accuracy_record():

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


def needle_refinement_accuracy_record():
    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    e_list = [0] * 3
    eListcur = [0] * 3
    angle_list = [0] * 2


    anchor = 0
    count = 0
    angle_count = 0


    while True:
        if Tis.Snap_image(1) is True:
            rawframe = Tis.Get_image()
            rawframe = rawframe[:, :, :3]
            dis_frame = np.array(rawframe)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)


            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y) and diamondCorners:
                    # m, b = line_polyfit(x, y)
                    # p1 = (0, round(b))
                    # p2 = (1000, round(m * 1000 + b))
                    # cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

                    coord_3D = []
                    coord_3D_rf, coord_2D_rf = edge_refinement_linear(gray_frame, x, y, plist)

                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)
                    # for j in range(3):
                    #     cx = round(coord_3D[j][0])
                    #     cy = round(coord_3D[j][1])
                    #     cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)

                    tip, end = scale_estimation_multi_mod(coord_3D[0], coord_3D[1], coord_3D[2], dlist[0], dlist[1],
                                                      mtx, tip_off)
                    tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0],
                                                      dlist[1], mtx, tip_off)

                    tip_a, end_a = pose_trans_needle(tvec, rvec)

                    error = error_calc_board(tip, anchor=anchor)
                    error_rf = error_calc_board(tip_rf, anchor=anchor)
                    error_ar = error_calc_board(tip_a, anchor=anchor)


                    e_list[0] += error
                    e_list[1] += error_rf
                    e_list[2] += error_ar


                    eListcur[0] = round(error, 2)
                    eListcur[1] = round(error_rf, 2)
                    eListcur[2] = round(error_ar, 2)

                    count += 1
                    eList = [round(x / count, 2) for x in e_list]
                    print(eListcur, eList, count)

                    # angle_diff, dist_diff = error_calc(tip_a, end_a, tip_rf, end_rf)
                    # print(round(angle_diff, 2), round(dist_diff, 2))
                    #
                    # angle_list[0] += angle_error
                    # angle_list[1] += angle_error_rf
                    # angle_count += 1
                    # angleList = [round(x / angle_count, 2) for x in angle_list]
                    # print(angleList, angle_count)

                    if error_rf > 1.5:
                        outputPath = '../All_images/error_investigate/different_keypoint'
                        outputPath_blank = '../All_images/error_investigate/different_keypoint_blank'
                        ts = datetime.datetime.now()
                        filename = "{}-{:.2f}.jpg".format(ts.strftime("%M-%S"), error_rf)

                        path = os.path.sep.join((outputPath, filename))
                        path_raw = os.path.sep.join((outputPath_blank, filename))

                        cv2.imwrite(path_raw, frame)

                        for j in range(3):

                            cv2.circle(frame, (coord_2D_rf[j][0], coord_2D_rf[j][1]), 1, (0, 255, 0), -1)
                            string = f"{coord_2D_rf[j][0]}, {coord_2D_rf[j][1]}"
                            cv2.putText(frame, string, (1000, 130 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 1, cv2.LINE_AA)

                        cv2.putText(frame, str(error_rf), (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(frame, f"{m:.2f}, {b:.2f}", (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1,
                                    cv2.LINE_AA)

                        cv2.imwrite(path, frame)
                        print('Record')




        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            anchor += 1
            print(f"Now point to hole {anchor}")


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def needle_refinement_precision_record():

    mtx, dist = camera_para_retrieve()
    for i in range(5):
        dis_frame = cv2.imread('../All_images/error_investigate/Blank/34-03-1.93.jpg')
        frame = undistort_img(dis_frame, mtx, dist)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)


        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]
            m, b = line_polyfit(x, y)
            print(m, b)
            if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y) and diamondCorners:
                coord_3D_rf, coord_2D_rf = edge_refinement_linear(gray_frame, x, y, plist)

                tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0], dlist[1], mtx, tip_off)
                tip_a, end_a = pose_trans_needle(tvec, rvec)

                print(tip_rf)




if __name__ == "__main__":
    # aruco_accuracy_record()
    # needle_refinement_accuracy_record()
    needle_refinement_precision_record()