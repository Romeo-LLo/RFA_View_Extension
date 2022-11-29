
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

plists = [[2, 4, 7], [1, 4, 8]]
dlists = [[38, 48], [48, 57.5]]
tip_offset = [3.2, 2.25]
# plist = [1, 4, 8]
# dlist = [48.262, 57.915]
# tip_off = 2.2
plist = [2, 4, 8]
dlist = [40, 60]
tip_off = 3.2


def arucoboard_test():
    camera_matrix, dist_coefs = camera_para_retrieve()
    dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    board = aruco.CharucoBoard_create(5, 4, 4.5, 3.5, dictionary)

    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)

    Tis.Start_pipeline()
    num_pt = 5
    avg_coord3D = np.zeros((num_pt, 3))
    frame_count = 0
    state = False
    while True and frame_count < 30:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            img_board = np.array(frame)
            img_color = img_board
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image=img_board,
                                                                     dictionary=dictionary,
                                                                     parameters=None,
                                                                     cameraMatrix=camera_matrix,
                                                                     distCoeff=dist_coefs)

            if corners:

                cv2.aruco.drawDetectedMarkers(image=img_color, corners=corners, ids=ids, borderColor=None)

                # 棋盘格黑白块内角点
                retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                                        markerIds=ids,
                                                                                        image=img_board,
                                                                                        board=board,
                                                                                        cameraMatrix=camera_matrix,
                                                                                        distCoeffs=dist_coefs)

                if np.any(charucoIds != None):

                    cv2.aruco.drawDetectedCornersCharuco(img_color, charucoCorners, charucoIds, [0, 0, 255])


                    rvec = None
                    tvec = None
                    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
                                                                           camera_matrix,
                                                                           dist_coefs, rvec, tvec)


                    if retval:
                        cv2.aruco.drawAxis(img_color, camera_matrix, dist_coefs, rvec, tvec, 5)
                        print(tvec)
                        if state:
                            coord3D = board_offset(rvec, tvec)
                            avg_coord3D += coord3D
                            print(coord3D)
                            frame_count += 1
            img_colorS = cv2.resize(img_color, (1080, 810))
            cv2.imshow("out", img_colorS)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                state = True
    avg_coord3D /= frame_count
    print("avg", avg_coord3D)
    np.save("../Coordinate/board_coordinate.npy", avg_coord3D)
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def realtime_error_snapshot():

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    est_state = False

    anchor = 1
    while True:
        if Tis.Snap_image(1) is True:

            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]


                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):

                    # for i in range(11):
                    #     cv2.circle(frame, (round(kp[0][i][0]), round(kp[0][i][1])), 1, (0, 255, 0), -1)
                    #     cv2.putText(frame, str(i), (round(kp[0][i][0]), round(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    #                 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                    m, b = line_polyfit(x, y)
                    p1 = (0, round(b))
                    p2 = (1000, round(m * 1000 + b))

                    # cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

                    coord_3D = []
                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                    tip, end = scale_estimation_multi(coord_3D[0], coord_3D[1], coord_3D[2], dlist[0], dlist[1],
                                                          mtx, tip_off)

                    error = error_calc_board(tip, anchor=anchor)
                    error_vec = error_vec_calc_board(tip, anchor=anchor)
                    print(f'{error:.2f}')

                    if est_state:

                        outputPath = '../All_images/edge_investigate/Blank'
                        ts = datetime.datetime.now()
                        filename = "{}-{:.2f}.jpg".format(ts.strftime("%M-%S"), error)
                        path = os.path.sep.join((outputPath, filename))

                        # cv2.putText(frame, str(error), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1,
                        #             cv2.LINE_AA)
                        # for i, pt in enumerate(plist):
                        #     coord = f"{kp[0][pt][0]:.2f} {kp[0][pt][1]:.2f}"
                        #     cv2.putText(frame, coord, (800, 150 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                        #                 cv2.LINE_AA)
                        # vec = f"{error_vec[0]} {error_vec[1]} {error_vec[2]}"
                        # pos = f"{tip[0]:.2f} {tip[1]:.2f} {tip[2]:.2f}"
                        #
                        # cv2.putText(frame, vec, (800, 150 + 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                        #             cv2.LINE_AA)
                        # cv2.putText(frame, pos, (800, 150 + 30 * (i + 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                        #             cv2.LINE_AA)
                        cv2.imwrite(path, frame)
                        print('Record')
                        est_state = False


            cv2.imshow('Window', frame)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                # esc
                est_state = True
            elif k == 13:
                # enter
                anchor += 1
                print(f"Now point to hole {anchor}")
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def realtime_error_board_refinement():
    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    e_list = [0] * 2
    eListcur = [0] * 2

    anchor = 1
    count = 0
    est_state = False

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
                    # for i in range(11):
                    #     cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 255, 0), -1)
                    #     cv2.puText(frame, str(i), (round(kp[0][i][0]), round(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #                 (0, 255, 0), 1, cv2.LINE_AA)

                    # m, b = line_polyfit(x, y)
                    # p1 = (0, round(b))
                    # p2 = (1000, round(m * 1000 + b))
                    # cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

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
                    error_vec = error_vec_calc_board(tip_rf, anchor=anchor)

                    e_list[0] += error
                    e_list[1] += error_rf

                    eListcur[0] = round(error, 2)
                    eListcur[1] = round(error_rf, 2)
                    count += 1
                    eList = [round(x / count, 2) for x in e_list]
                    print(eListcur, eList, count)
                    # print(eListcur, error_vec[2]/error_rf)


                    if est_state:

                        outputPath = '../All_images/edge_investigate/Refine_linear'
                        ts = datetime.datetime.now()
                        filename = "{}-{:.2f}.jpg".format(ts.strftime("%M-%S"), error)
                        path = os.path.sep.join((outputPath, filename))
                        cv2.putText(frame, str(round(error, 2)), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, str(round(error_rf, 2)), (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                        vec = f"{error_vec[0]} {error_vec[1]} {error_vec[2]}"
                        cv2.putText(frame, vec, (800, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                        for coord in coord_2D_rf:
                            frame[coord[1]][coord[0]] = (0, 255, 0)

                        cv2.imwrite(path, frame)
                        print('Record')
                        est_state = False


        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            anchor += 1
            print(f"Now point to hole {anchor}")
        elif k == 27:
            est_state = True

    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    arucoboard_test()
    # realtime_error_board_refinement()
    # realtime_error_snapshot()
