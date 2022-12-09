
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
import time

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

plists = [[2, 4, 8], [1, 5, 7]]
dlists = [[40 / lazer, 60 / lazer], [60 / lazer, 40 / lazer]]
tip_offset = [3.2, 2.3]

plist = [1, 4, 8]
dlist = [48.262, 57.915]
tip_off = 2.2
# plist = [2, 4, 8]
# dlist = [40, 60]
# tip_off = 3.2


def arucoboard_test():
    camera_matrix, dist_coefs = camera_para_retrieve()
    dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    board = aruco.CharucoBoard_create(5, 4, 4.5, 3.5, dictionary)

    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)

    Tis.Start_pipeline()
    num_pt = 10
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
                    # m, b = line_polyfit(x, y)
                    # p1 = (0, round(b))
                    # p2 = (1000, round(m * 1000 + b))

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

                        outputPath = '../All_images/error_investigate/Blank'
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


def realtime_refinement_visual():

    plt.ion()
    fig = plt.figure()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.65,
                        hspace=0.4)
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, aspect='equal')

    num_lines = 2
    num_steps = 2

    color = ['red', 'green']
    lines = [ax.plot([], [], [], color[i], label='Needle trajectory')[0] for i in range(num_lines)]
    lines2 = [ax2.plot([], [], color[i])[0] for i in range(num_lines)]

    sq_len = 15
    deep = 70
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-sq_len, sq_len)
    ax.set_ylim(40, deep)
    ax.set_zlim(-sq_len, sq_len)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    us_zone = 17
    us_x, us_z = np.meshgrid(range(-us_zone, us_zone), range(-us_zone, us_zone))
    us_y = np.full((2*us_zone, 2*us_zone), deep)
    ax.plot_surface(us_x, us_y, us_z, alpha=0.5, label='Ultrasound plane')

    sq_len = 30
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(-sq_len, sq_len)
    ax2.set_ylim(-sq_len, sq_len)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])


    us_img = plt.imread('../All_images/ultrasound_show.png')
    ax2.imshow(us_img, extent=[-sq_len, sq_len, -sq_len, sq_len])

    smooth = True
    sm_factor = 0.8

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    anchor = 1
    first = True

    while True:
        if Tis.Snap_image(1) is True:
            trajs = np.zeros((num_lines, num_steps, 3))
            trajs2 = np.zeros((num_lines, num_steps, 2))

            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)
            if diamondCorners:
                tip_a, end_a = pose_trans_needle(tvec, rvec)
                trajs[0] = np.array([tip_a, end_a])
                trajs2[0] = np.array([tip_a[:2], end_a[:2]])

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints

            if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):

                    # m, b = line_polyfit(x, y)
                    # p1 = (0, round(b))
                    # p2 = (1500, round(m * 1500 + b))
                    # cv2.line(frame, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

                    for i in range(11):
                        cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 5, (0, 255, 0), -1)

                    avg_tip = np.empty((0, 3), float)
                    for i, (ps, ds) in enumerate(zip(plists, dlists)):
                        coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod(gray_frame, x, y, ps)

                        tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], ds[0],
                                                          ds[1], mtx, tip_offset[i])
                        avg_tip = np.append(avg_tip, np.array([tip_rf]), axis=0)


                    tip_diff = round(np.linalg.norm(avg_tip[0] - avg_tip[1]), 2)
                    # if tip_diff < 1:
                    #     trajs[1] = np.array([tip_rf, end_rf])
                    #     trajs2[1] = np.array([tip_rf[:2], end_rf[:2]])

                    # if first:
                    #     trajs[1] = np.array([tip_rf, end_rf])
                    #     first = False
                    # else:
                    #     sm_tip = trajs[1][0] * sm_factor + tip_rf * (1 - sm_factor)
                    #     sm_end = trajs[1][1] * sm_factor + end_rf * (1 - sm_factor)
                    #     trajs[1] = np.array([sm_tip, sm_end])

                    if diamondCorners:
                        angle_diff, dist_diff = error_calc(tip_a, end_a, tip_rf, end_rf)
                        # print(round(angle_diff, 2), round(dist_diff, 2))

            for line, traj in zip(lines, trajs):
                line.set_data(traj[:, 0], traj[:, 2])
                line.set_3d_properties(-traj[:, 1])

            for line, traj in zip(lines2, trajs2):
                line.set_data(-traj[:, 0], -traj[:, 1])


        fig.canvas.draw()
        fig.canvas.flush_events()
        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            plt.savefig("../All_images/demo.jpg")
            cv2.imwrite("../All_images/demo2.jpg", frame)
            print('Record')
            # anchor += 1
            # print(f"Now point to hole {anchor}")


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # arucoboard_test()
    realtime_refinement_visual()
    # realtime_error_snapshot()
