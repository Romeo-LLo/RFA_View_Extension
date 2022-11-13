
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
import datetime

gi.require_version("Gst", "1.0")
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_final3.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)




def line_fit():

    num_steps = 5
    mtx, dist = camera_para_retrieve()
    frame_num = 0
    avg_angle_error = 0
    avg_dist_error = 0
    avg_angle_error_fit = 0
    avg_dist_error_fit = 0
    for id in range(1, 6):
        cap = cv2.VideoCapture(f'../All_images/video{id}.mp4')
        while cap.isOpened():
            ret, dis_frame = cap.read()
            if ret:
                frame = undistort_img(dis_frame, mtx, dist)
                diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)

                outputs = predictor(frame)
                kp_tensor = outputs["instances"].pred_keypoints
                if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
                    continue

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]
                trajs = np.zeros((2, num_steps, 3))

                if isMonotonic(x) and isMonotonic(y):
                    coord_3D = []
                    plist = [2, 4, 6]

                    coord_3D_fit = []
                    kp_fit = orth_fit(x, y)
                    # coord_3D_fit = partial_orth_fit(x, y, plist)

                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                        pt_f = np.array([kp_fit[0][i][0], kp_fit[0][i][1], 0], dtype='float64')
                        coord_3D_fit.append(pt_f)


                    tip, end = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], 40, 40, mtx)
                    tip_f, end_f = scale_estimation(coord_3D_fit[0], coord_3D_fit[1], coord_3D_fit[2], 40, 40, mtx)

                    trajs[1] = np.linspace(tip, end, num=num_steps)

                    if diamondCorners:
                        tip_t = pose_trans_needle(tvec, rvec, 21.2)
                        end_t = pose_trans_needle(tvec, rvec, 3)
                        trajs[0] = np.linspace(tip_t, end_t, num=num_steps)

                        frame_num += 1
                        angle_error, dist_error = error_calc(tip_t, end_t, tip, end)
                        avg_angle_error += angle_error
                        avg_dist_error += dist_error
                        # print('normal', angle_error, dist_error)

                        angle_error_fit, dist_error_fit = error_calc(tip_t, end_t, tip_f, end_f)
                        avg_angle_error_fit += angle_error_fit
                        avg_dist_error_fit += dist_error_fit
                        # print('fit', angle_error_fit, dist_error_fit)

                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        cap.release()
        print(f'{frame_num} for video{id}')

    print(f'Avg angle err = {avg_angle_error / frame_num}')
    print(f'Avg dist err = {avg_dist_error / frame_num}')

    print(f'Avg angle err fit= {avg_angle_error_fit / frame_num}')
    print(f'Avg dist err fit = {avg_dist_error_fit / frame_num}')


def line_multi():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_steps = 5
    num_lines = 4
    color = ['red', 'green', 'blue', 'yellow']

    lines = [ax.plot([], [], [], color[i])[0] for i in range(num_lines)]
    sq_len = 20
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-sq_len, sq_len)
    ax.set_ylim(55, 75)
    ax.set_zlim(-sq_len, sq_len)

    mtx, dist = camera_para_retrieve()
    frame_num = 0
    success_frame = 0

    avg_angle_error_mult = 0
    avg_dist_error_mult = 0
    for id in range(1, 6):
        cap = cv2.VideoCapture(f'../All_images/video{id}.mp4')
        while cap.isOpened():
            ret, dis_frame = cap.read()
            if ret:
                frame_num += 1
                frame = undistort_img(dis_frame, mtx, dist)
                diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)

                outputs = predictor(frame)
                kp_tensor = outputs["instances"].pred_keypoints
                if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
                    continue

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                trajs = np.zeros((num_lines, num_steps, 3))

                if isMonotonic(x) and isMonotonic(y):
                    for i in range(11):
                        cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
                        cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)

                    plists = [[2, 4, 6], [1, 5, 7], [5, 8, 9]]
                    dlists = [[40, 40], [60, 40], [50, 30]]
                    tip_offset = [3.2, 2.2, 8.2]

                    avg_tip = np.empty((0, 3), float)
                    avg_end = np.empty((0, 3), float)
                    for j, plist in enumerate(plists):
                        coord_3D = []
                        for i in plist:
                            pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                            coord_3D.append(pt)

                        tip, end = scale_estimation_multi(coord_3D[0], coord_3D[1], coord_3D[2], dlists[j][0], dlists[j][1], mtx, tip_offset[j])
                        avg_tip = np.append(avg_tip, np.array([tip]), axis=0)
                        avg_end = np.append(avg_end, np.array([end]), axis=0)
                        # trajs[j] = np.linspace(tip, end, num=num_steps)

                    tip_std = np.std(avg_tip, axis=0)
                    end_std = np.std(avg_end, axis=0)
                    success_frame += 1

                    if tip_std[2] < 5 and end_std[2] < 5 and diamondCorners:
                        tip = np.mean(avg_tip, axis=0)
                        end = np.mean(avg_end, axis=0)
                        trajs[0] = np.linspace(tip, end, num=num_steps)
                        # success_frame += 1
                    if diamondCorners:
                        tip_t = pose_trans_needle(tvec, rvec, 21.2)
                        end_t = pose_trans_needle(tvec, rvec, 3)
                        trajs[-1] = np.linspace(tip_t, end_t, num=num_steps)

                        # frame_num += 1
                        # angle_error, dist_error = error_calc(tip_t, end_t, tip, end)
                        # avg_angle_error_mult += angle_error
                        # avg_dist_error_mult += dist_error

                for line, traj in zip(lines, trajs):
                    line.set_data(traj[:, 0], traj[:, 2])
                    line.set_3d_properties(-traj[:, 1])

                fig.canvas.draw()
                fig.canvas.flush_events()
                frameS = cv2.resize(frame, (900, 675))
                cv2.imshow('Window', frameS)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        cap.release()
        print(f'{success_frame}/{frame_num} for video{id}')

        # print(f'{frame_num} for video{id}')

    # print(f'Avg angle err = {avg_angle_error_mult / frame_num}')
    # print(f'Avg dist err = {avg_dist_error_mult / frame_num}')

    # print(f'Avg angle err fit= {avg_angle_error_fit / frame_num}')
    # print(f'Avg dist err fit = {avg_dist_error_fit / frame_num}')


def line_multi_video():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_steps = 10
    num_lines = 5
    color = ['red', 'green', 'blue', 'purple', 'orange']

    lines = [ax.plot([], [], [], color[i])[0] for i in range(num_lines)]
    sq_len = 20
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-sq_len, sq_len)
    ax.set_ylim(55, 75)
    ax.set_zlim(-sq_len, sq_len)

    mtx, dist = camera_para_retrieve()
    cap = cv2.VideoCapture(f'../All_images/video5.mp4')
    while cap.isOpened():
        ret, dis_frame = cap.read()
        if ret:
            frame = undistort_img(dis_frame, mtx, dist)
            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) == 0 or torch.isnan(kp_tensor).any():
                continue

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            trajs = np.zeros((num_lines, num_steps, 3))

            if isMonotonic(x) and isMonotonic(y) and diamondCorners:
                tip_t, end_t = pose_trans_needle(tvec, rvec)
                trajs[-1] = np.linspace(tip_t, end_t, num=num_steps)

                for i in range(11):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 1, cv2.LINE_AA)

                # plists = [[1, 4, 7], [1, 4, 8], [2, 5, 8]]
                # dlists = [[50, 50], [50, 60], [50, 50]]
                # tip_offset = [2.2, 2.2, 3.2]
                plists = [[1, 4, 8]]
                dlists = [[50, 60]]
                tip_offset = [2.2]

                avg_tip = np.empty((0, 3), float)
                avg_end = np.empty((0, 3), float)
                for j, plist in enumerate(plists):
                    coord_3D = []
                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                    tip, end = scale_estimation_multi(coord_3D[0], coord_3D[1], coord_3D[2], dlists[j][0], dlists[j][1], mtx, tip_offset[j])
                    trajs[j] = np.linspace(tip, end, num=num_steps)
                    # angle_error, dist_error = error_calc(tip_t, end_t, tip, end)
                    # print(f'For points {plist} / angle: {angle_error}, dist: {dist_error}')
                    avg_tip = np.append(avg_tip, np.array([tip]), axis=0)
                    avg_end = np.append(avg_end, np.array([end]), axis=0)


                tip = np.mean(avg_tip, axis=0)
                end = np.mean(avg_end, axis=0)
                trajs[j+1] = np.linspace(tip, end, num=num_steps)
                print(tip_t, tip)
                # print(end_t, end)
                angle_error, dist_error = error_calc(tip_t, end_t, tip, end)
                print(f'For average / angle: {angle_error:.2f}, dist: {dist_error:.2f}')




                for line, traj in zip(lines, trajs):
                    line.set_data(traj[:, 0], traj[:, 2])
                    line.set_3d_properties(-traj[:, 1])

                fig.canvas.draw()
                fig.canvas.flush_events()
                frameS = cv2.resize(frame, (900, 675))
                cv2.imshow('Window', frameS)
                cv2.waitKey(0)
            else:
              continue
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    cap.release()

def error_test_frame():
    interval = [22, 10, 30, 10, 10, 30, 10, 10, 30, 10]
    plists = [[2, 4, 6], [1, 5, 7], [1, 4, 7], [5, 8, 9], [4, 6, 8], [2, 5, 8], [3, 5, 8], [1, 3, 5], [1, 4, 8], [2, 5, 7]]
    dlists = [[40, 40], [60, 40], [50, 50], [50, 30], [40, 20], [50, 50], [20, 50], [40, 20], [50, 60], [50, 40]]
    tip_offset = [3.2, 2.2, 2.2, 8.2, 7.2, 3.2, 6.2, 2.2, 2.2, 3.2]

    dist_error_list = np.zeros((len(plists)))
    angle_error_list = np.zeros((len(plists)))
    frame_num = 0
    mtx, dist = camera_para_retrieve()
    for order in range(1, 6):
        cap = cv2.VideoCapture(f'../All_images/video{order}.mp4')

        while cap.isOpened():
            ret, dis_frame = cap.read()
            if ret:
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
                    frame_num += 1
                    tip_t, end_t = pose_trans_needle(tvec, rvec)

                    for j, plist in enumerate(plists):
                        coord_3D = []
                        for i in plist:
                            pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                            coord_3D.append(pt)
                        tip, end = scale_estimation_multi(coord_3D[0], coord_3D[1], coord_3D[2], dlists[j][0], dlists[j][1], mtx, tip_offset[j])
                        angle_error, dist_error = error_calc(tip_t, end_t, tip, end)
                        print(angle_error, dist_error)
                        angle_error_list[j] += angle_error
                        dist_error_list[j] += dist_error
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        cap.release()

    angle_error_list /= frame_num
    dist_error_list /= frame_num
    print(angle_error_list)
    print(dist_error_list)
    print(frame_num)

    # Result
    #angle_error_list [12.9361107  11.16901311  9.38663259 18.2396625  22.17041563  8.92488851 22.06760178 23.31326559  7.13176133 12.75986461]
    #dist_error_list [3.78392626 3.07097142 3.07152368 4.86935637 6.72194453 2.80038621 6.26900182 8.82567047 2.7577783  3.20275442]




def aruco_test():
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
            if diamondCorners:
                aruco.drawAxis(frame, mtx, dist, rvec, tvec, 3)

            if diamondCorners:
                tip_t, end_t = pose_trans_needle(tvec, rvec)
                t = tvec[0][0]
                print(f"[{t[0]:.2f} {tip_t[0]:.2f}] [{t[1]:.2f} {tip_t[1]:.2f}] [{t[2]:.2f} {tip_t[2]:.2f}] {tip_t[2]-t[2]:.2f}")
            cv2.imshow('Window', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def arucoboard_test():
    camera_matrix, dist_coefs = camera_para_retrieve()
    dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    # board = aruco.CharucoBoard_create(9, 5, 3, 2.3, dictionary)
    board = aruco.CharucoBoard_create(5, 4, 4.5, 3.5, dictionary)

    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)

    Tis.Start_pipeline()
    num_pt = 5
    avg_coord3D = np.zeros((num_pt, 3))
    frame_count = 0
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
                        coord3D = board_offset(rvec, tvec)
                        avg_coord3D += coord3D
                        print(coord3D)
                        frame_count += 1
            cv2.imshow("out", img_color)
            if cv2.waitKey(1) == ord('q'):
                break
    avg_coord3D /= frame_count
    print("avg", avg_coord3D)
    np.save("../Coordinate/board_coordinate.npy", avg_coord3D)
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def realtime_error_board():

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()


    while True:
        if Tis.Snap_image(1) is True:
            est_state = False

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
                tip_t, end_t = pose_trans_needle(tvec, rvec)
                # error = error_calc_board(tip_t, 0)

            if isMonotonic(x) and isMonotonic(y):
                for i in range(11):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 1, cv2.LINE_AA)

                plists = [[2, 5, 8]]
                dlists = [[50, 50]]
                tip_offset = [3.2]

                avg_tip = np.empty((0, 3), float)
                avg_end = np.empty((0, 3), float)
                for j, plist in enumerate(plists):
                    coord_3D = []
                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                    tip, end = scale_estimation_multi(coord_3D[0], coord_3D[1], coord_3D[2], dlists[j][0], dlists[j][1],
                                                      mtx, tip_offset[j])

                    avg_tip = np.append(avg_tip, np.array([tip]), axis=0)
                    avg_end = np.append(avg_end, np.array([end]), axis=0)

                tip = np.mean(avg_tip, axis=0)
                end = np.mean(avg_end, axis=0)
                error = error_calc_board(tip, 1)
                est_state = True



        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        if est_state:
            if error < 1:
                outputPath = '../All_images/error_investigate'
                ts = datetime.datetime.now()
                filename = "{}.jpg".format(ts.strftime("%M-%S"))
                path = os.path.sep.join((outputPath, filename))

                cv2.putText(frame, str(error), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                for i, pt in enumerate(plists[0]):
                    coord = f"{kp[0][pt][0]} {kp[0][pt][1]}"
                    cv2.putText(frame, coord, (800, 120 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                                cv2.LINE_AA)

                cv2.imwrite(path, frame)
                print('Good')

        k = cv2.waitKey(30) & 0xFF
        if k == 27 and est_state:
            outputPath = '../All_images/error_investigate'
            ts = datetime.datetime.now()
            filename = "{}.jpg".format(ts.strftime("%M-%S"))
            path = os.path.sep.join((outputPath, filename))

            cv2.putText(frame, str(error), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            for i, pt in enumerate(plists[0]):
                coord = f"{kp[0][pt][0]} {kp[0][pt][1]}"
                cv2.putText(frame, coord, (800, 120+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imwrite(path, frame)
            print('Capture')

    Tis.Stop_pipeline()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # aruco_test()
    # arucoboard_test()
    realtime_error_board()
    # line_multi_video()
    # error_test_frame()
  # camera_para_retrieve()[ -0.0484527  -20.0817824   -2.91639071]z