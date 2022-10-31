
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import TIS
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from needle_utils import *
from Linear_equation import *
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'model_final3.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)


import time
import sys
import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst

def video():
    mtx, dist = camera_para_retrieve()
    cap = cv2.VideoCapture('../All_images/needle_test0915.mp4')
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        frame = undistort_img(frame, mtx, dist)
        if not ret:
            print('Fail')
            break

        outputs = predictor(frame)
        diamondCorners, rvec, tvec = diamond_detection(frame, mtx, dist)

        kp = outputs["instances"].pred_keypoints.to("cpu").numpy()
        if kp.shape[0] != 0:
            # for i in range(10):
            #     cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
            #     cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
            #                 (0, 0, 255), 1, cv2.LINE_AA)

            coord_3D = []
            for i in range(3):
                pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                coord_3D.append(pt)

            est_tvec = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], 32, 30, mtx)

            if diamondCorners != None:
                print(int(tvec[0][0][2]), int(est_tvec[2]))

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            # frameS = cv2.resize(frame, (720, 540))
            # cv2.imshow('Window', frameS)
        end = time.time()
        print(1 / (end - start))
    cap.release()
    cv2.destroyAllWindows()
    print('Program ends')


def frame():
    mtx, dist = camera_para_retrieve()
    cap = cv2.VideoCapture('../All_images/needle_test0915.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        frame = undistort_img(frame, mtx, dist)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diamondCorners, rvec, tvec = diamond_detection(frame, mtx, dist)

        if not ret:
            print('Fail')
            break

        outputs = predictor(frame)

        kp = outputs["instances"].pred_keypoints.to("cpu").numpy()
        if kp.shape[0] != 0:

            fit_kp = line_fit(kp)


            corners = np.array(kp[0, :-1, :2])
            corners = np.float32(corners.astype(int))
            rf_corners = corner_refinement(gray_frame, corners)
            for i in range(10):
                cv2.circle(frame, (round(kp[0][i][0]), round(kp[0][i][1])), 3, (0, 255, 0), -1)
                # cv2.circle(frame, (int(rf_corners[i][0]), int(rf_corners[i][1])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (round(fit_kp[i][0]), round(fit_kp[i][1])), 2, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)

            coord_3D = []
            rf_coord_3D = []

            for i in range(1, 4):
                pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                coord_3D.append(pt)
                rf_pt = np.array([rf_corners[i][0], rf_corners[i][1], 0], dtype='float64')
                rf_coord_3D.append(rf_pt)

            est_tvec = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], 30, 10, mtx)
            rf_est_tvec = scale_estimation(rf_coord_3D[0], rf_coord_3D[1], rf_coord_3D[2], 30, 10, mtx)

            if diamondCorners != None:
                trans_tvec = pose_trans_needle(tvec, rvec) #translation from marker to needle tip
                print(trans_tvec[0][2], est_tvec[2], rf_est_tvec[2])
                frameS = cv2.resize(frame, (900, 675))
                cv2.imshow('Window', frameS)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

def test_algorithm():
    mtx, dist = camera_para_retrieve()
    cap = cv2.VideoCapture('../All_images/needle_test0915.mp4')

    count = 0
    error = 0
    while cap.isOpened():

        ret, frame = cap.read()
        frame = undistort_img(frame, mtx, dist)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diamondCorners, rvec, tvec = diamond_detection(frame, mtx, dist)
        if diamondCorners == None:
            continue

        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
            continue

        kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
        x = kp[0, :-1, 0]
        y = kp[0, :-1, 1]

        if not isMonotonic(x) and not isMonotonic(y):
            continue
        fit_kp = line_fit(kp[:, :-1, :])

        corners = np.array(fit_kp[0])
        corners = np.float32(corners.astype(int))
        rf_corners = corner_refinement(gray_frame, corners)
        rf_corners = np.expand_dims(rf_corners, axis=0)

        # for i in range(10):
        #     cv2.circle(frame, (round(kp[0][i][0]), round(kp[0][i][1])), 3, (0, 255, 0), -1)
        #     cv2.circle(frame, (round(rf_corners[0][i][0]), round(rf_corners[0][i][1])), 3, (255, 0, 0), -1)
        #     # cv2.circle(frame, (round(fit_kp[0][i][0]), round(fit_kp[0][i][1])), 1, (0, 0, 255), -1)

        #     cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (255, 0, 0), 1, cv2.LINE_AA)


        # plist = [[0, 1, 2], [1, 2, 3], [1, 2, 5]]
        # llist = [[32, 30], [30, 10], [30, 50]]
        plist = [[1, 2, 5]]
        llist = [[30, 50]]
        for j, p in enumerate(plist):

            coord_3D = []
            rf_coord_3D = []
            fit_coord_3D = []


            for i in range(3):
                pt = np.array([kp[0][p[i]][0], kp[0][p[i]][1], 0], dtype='float64')
                coord_3D.append(pt)
                rf_pt = np.array([rf_corners[0][p[i]][0], rf_corners[0][p[i]][1], 0], dtype='float64')
                rf_coord_3D.append(rf_pt)
                fit_pt = np.array([fit_kp[0][p[i]][0], fit_kp[0][p[i]][1], 0], dtype='float64')
                fit_coord_3D.append(fit_pt)

            est_tvec = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], llist[j][0], llist[j][1], mtx)
            rf_est_tvec = scale_estimation(rf_coord_3D[0], rf_coord_3D[1], rf_coord_3D[2], llist[j][0], llist[j][1], mtx)
            fit_est_tvec = scale_estimation(fit_coord_3D[0], fit_coord_3D[1], fit_coord_3D[2], llist[j][0], llist[j][1], mtx)

            trans_tvec = pose_trans_needle(tvec, rvec) #translation from marker to needle tip
            trans_error = np.linalg.norm(trans_tvec - est_tvec)
            # cur_error = np.array([[est_tvec, rf_est_tvec, fit_est_tvec]]) - trans_tvec
            # cur_error = np.absolute(cur_error)
            error += trans_error
            print(trans_tvec, est_tvec)
            count += 1


            frameS = cv2.resize(frame, (900, 675))
            cv2.imshow('Window', frameS)
            cv2.waitKey(0)
    print('avg', error / count)
    cap.release()
    cv2.destroyAllWindows()

def realtime():
    import sys
    # sys.path.append("../python-common")

    mtx, dist = camera_para_retrieve()

    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)

    Tis.Start_pipeline()

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
                continue

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            if isMonotonic(x) and isMonotonic(y) and diamondCorners:
                for i in range(11):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

                coord_3D = []
                # plist = [1, 2, 4, 5]
                plist = [2, 4, 6]

                for i in plist:
                    pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                    coord_3D.append(pt)

                est_tvec = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], 40, 40, mtx)
                # est_tvec = scale_estimation_4p(coord_3D[0], coord_3D[1], coord_3D[2], coord_3D[3], 30, 20, 30, mtx)
                trans_tvec = pose_trans_needle(tvec, rvec)  # translation from marker to needle tip
                error = np.linalg.norm(trans_tvec - est_tvec)
                # print('gt', transSvec)
                print('error', error)

            frameS = cv2.resize(frame, (900, 675))
            cv2.imshow('Window', frameS)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def realtime_draw_pts():

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)


            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
                continue

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]
            #
            if isMonotonic(x) and isMonotonic(y):
                for i in range(11):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

            frameS = cv2.resize(frame, (900, 675))

            cv2.imshow('Window', frameS)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()
    print('Program ends')


def realtime_visual():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([], [], [])[0] for _ in range(2)]
    sq_len = 10
    deep = 80
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-sq_len, sq_len)
    ax.set_ylim(40, deep)
    ax.set_zlim(-sq_len, sq_len)
    num_steps = 10

    # plt.gca().invert_yaxis()
    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)


            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
                continue

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            if diamondCorners:
                p1t = pose_trans_needle(tvec, rvec, 18)
                p4t = pose_trans_needle(tvec, rvec, 3)
                if isMonotonic(x) and isMonotonic(y):
                    for i in range(11):
                        cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
                        cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                    coord_3D = []
                    plist = [2, 4, 6]

                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                    p1, p4 = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], 40, 40, mtx)

                    trajs = np.zeros((2, num_steps, 3))
                    trajs[0] = np.linspace(p1t, p4t, num=num_steps)
                    trajs[1] = np.linspace(p1, p4, num=num_steps)
                    for line, traj in zip(lines, trajs):
                        line.set_data(traj[:, 0], traj[:, 2])
                        line.set_3d_properties(-traj[:, 1])
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                else:
                    trajs = np.zeros((2, num_steps, 3))
                    trajs[0] = np.linspace(p1t, p4t, num=num_steps)
                    for line, traj in zip(lines, trajs):
                        line.set_data(traj[:, 0], traj[:, 2])
                        line.set_3d_properties(-traj[:, 1])
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            else:
                trajs = np.zeros((2, num_steps, 3))
                for line, traj in zip(lines, trajs):
                    line.set_data(traj[:, 0], traj[:, 2])
                    line.set_3d_properties(-traj[:, 1])
                fig.canvas.draw()
                fig.canvas.flush_events()

            frameS = cv2.resize(frame, (900, 675))

            cv2.imshow('Window', frameS)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_visual()
    # realtime()
    # realtime_draw_pts()
    # video()
    # frame()
    # test_algorithm()