
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
from partial_filter import *
from edge_refinement import *
import torch
import gi
import cv2.aruco as aruco
import datetime
import time

from collections import deque
from statistics import mean, stdev

mtx, dist = camera_para_retrieve()

gi.require_version("Gst", "1.0")
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_1227.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set aq custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)

lazer = 1.036

def aruco_accuracy_record():

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    avg_error = 0
    frame_count = 0
    anchor = 0
    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            frame = np.array(frame)
            # frame = undistort_img(dis_frame, mtx, dist)

            diamondCorners, rvec, tvec = diamond_detection(frame, mtx, dist)

            if diamondCorners:
                tip_a, end_a = pose_trans_needle(tvec, rvec)
                # error_vec = error_vec_calc_board(tip_a, anchor)
                # error = error_calc_board(tip_a, anchor)

                error = error_angle_board(tip_a, end_a, anchor=anchor)
                # print(f'{error:.2f}')q
                # if error < 1.5:
                cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)
                avg_error += error
                frame_count += 1

                print(round(error, 2), round(avg_error / frame_count, 2), frame_count)


        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)
        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            anchor += 1
            print(f"Now point to hole {anchor}")

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

    plist = [2, 4, 8]
    dlist = [40, 60]

    dlist = [d / lazer for d in dlist]
    tip_off = 3.2

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
                    coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod(gray_frame, x, y, plist)

                    for i in plist:
                        pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                        coord_3D.append(pt)

                    for coord in coord_2D_rf:
                        cv2.circle(frame, (coord[0], coord[1]), 1, (0, 255, 0), -1)

                    tip, end = scale_estimation_multi_mod(coord_3D[0], coord_3D[1], coord_3D[2], dlist[0],
                                                      dlist[1], mtx, tip_off)
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

                    # if error_rf > 2:
                    #
                    #     for i in range(3):
                    #         cv2.circle(frame, (coord_2D_rf[i][0], coord_2D_rf[i][1]), 1, (0, 0, 255), -1)
                    #
                    #     outputPath = '../All_images/error_investigate/different_keypoint'
                    #     ts = datetime.datetime.now()
                    #     filename = "{}-{:.2f}.jpg".format(ts.strftime("%M-%S"), error_rf)
                    #     path = os.path.sep.join((outputPath, filename))
                    #     cv2.imwrite(path, frame)
                    #     print('record')

                    # angle_diff, dist_diff = error_calc(tip_a, end_a, tip_rf, end_rf)
                    # print(round(angle_diff, 2), round(dist_diff, 2))
                    #
                    # angle_list[0] += angle_error
                    # angle_list[1] += angle_error_rf
                    # angle_count += 1
                    # angleList = [round(x / angle_count, 2) for x in angle_list]
                    # print(angleList, angle_count)


        frameS = cv2.resize(frame, (1080, 810))
        cv2.imshow('Window', frameS)

        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            anchor += 1
            print(f"Now point to hole {anchor}")


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()



def multi_set_needle_refinement_accuracy_record():
    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    num = 7 + 1
    e_list = [0] * num
    eListcur = [0] * num


    plists = [[2, 4, 8], [1, 5, 7], [1, 4, 6], [2, 4, 6], [3, 5, 8], [1, 3, 5], [1, 2, 4]]
    dlists = [[40 / lazer, 60 / lazer], [60 / lazer, 40 / lazer], [50 / lazer, 40 / lazer],
              [40 / lazer, 40 / lazer], [20 / lazer, 50 / lazer], [40 / lazer, 20 / lazer], [10 / lazer, 40 / lazer]]

    tip_offset = [3.2, 2.3, 2.3, 3.2, 6.1, 2.3, 2.3]

    anchor = 0
    count = 0
    est_state = False
    while True:
        if Tis.Snap_image(1) is True:
            rawframe = Tis.Get_image()
            rawframe = rawframe[:, :, :3]
            dis_frame = np.array(rawframe)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):

                    avg_tip = np.empty((0, 3), float)
                    for i, (ps, ds) in enumerate(zip(plists, dlists)):
                        coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod(gray_frame, x, y, ps)

                        for coord in coord_2D_rf:
                            cv2.circle(frame, (coord[0], coord[1]), 3, (0, 255, 0), -1)
                        plist = [2, 4, 8]
                        dlist = [40, 60]
                        tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0],
                                                          dlist[1], mtx, tip_offset[i])
                        avg_tip = np.append(avg_tip, np.array([tip_rf]), axis=0)
                        error_rf = error_calc_board(tip_rf, anchor=anchor)
                        # error_rf = error_angle_board(tip_rf, end_rf, anchor=anchor)
                        eListcur[i] = round(error_rf, 2)


                    tip_diff = round(np.linalg.norm(avg_tip[0] - avg_tip[1]), 2)

                    if tip_diff < 0.7:
                    # if True:
                        count += 1
                        for i in range(num):
                            e_list[i] += eListcur[i]
                        eList = [round(x / count, 2) for x in e_list]
                        print(eListcur, eList, count)


                    if est_state:
                        final = [x / count for x in e_list]
                        print(final, count)


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



def multi_set_needle_refinement_accuracy_video():
    mtx, dist = camera_para_retrieve()

    num = 5
    e_list = [0] * num
    eListcur = [0] * num

    plists = [[2, 4, 8], [1, 5, 7], [1, 2, 4], [1, 4, 6], [1, 3, 5]]
    dlists = [[40 / lazer, 60 / lazer], [60 / lazer, 40 / lazer], [10 / lazer, 40 / lazer], [50 / lazer, 40 / lazer], [40 / lazer, 20 / lazer]]
    tip_offset = [3.2, 2.2, 2.2, 2.2, 2.2]

    count = 0
    for anchor in range(4):
        images = glob.glob(f'../All_images/record/hole{anchor}/*.jpg')
        for img in images:
            dis_frame = cv2.imread(img)
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
                    avg_tip = np.empty((0, 3), float)
                    for i, (ps, ds) in enumerate(zip(plists, dlists)):
                        coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod(gray_frame, x, y, ps)

                        for coord in coord_2D_rf:
                            cv2.circle(frame, (coord[0], coord[1]), 1, (0, 255, 0), -1)

                        tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], ds[0],
                                                                    ds[1], mtx, tip_offset[i])
                        avg_tip = np.append(avg_tip, np.array([tip_rf]), axis=0)
                        error_rf = error_calc_board(tip_rf, anchor=anchor)
                        # error_vec = error_vec_calc_board(tip_rf, anchor=anchor)
                        # vec = [x + y for x, y in zip(vec, error_vec)]
                        eListcur[i] = round(error_rf, 2)


                    #
                    # tip_diff = round(np.linalg.norm(avg_tip[0] - avg_tip[1]), 2)
                    # tip_diff2 = round(np.linalg.norm(avg_tip[0] - avg_tip[3]), 2)

                    # if tip_diff < 0.3 and tip_diff2 < 0.3:
                    count += 1
                    for i in range(num):
                        e_list[i] += eListcur[i]
                    eList = [round(x / count, 2) for x in e_list]
                    print(eListcur, eList, count)

                    # outputPath = '../All_images/error_investigate/different_keypoint'
                    # ts = datetime.datetime.now()
                    # filename = "{}-{:.2f}-{:.2f}.jpg".format(ts.strftime("%M-%S"), eListcur[0], eListcur[1])
                    # path = os.path.sep.join((outputPath, filename))
                    # cv2.imwrite(path, frame)



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


def model_keypoint_error():
    mtx, dist = camera_para_retrieve()
    root = '../All_images/Testimg_undist_labeled'
    avg_err = 0
    avg_rf_err = 0
    count = 0
    with open('../All_images/Testimg_undist_labeled/train.json', newline='') as jsonfile:
        test = json.load(jsonfile)
        imgs = test['images']
        for i in range(len(imgs)):
            img_path = os.path.join(root, imgs[i]['file_name'])
            frame = cv2.imread(img_path)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():
                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                xs = kp[0, :-1, 0]
                ys = kp[0, :-1, 1]

                ps = list(range(11))
                coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, xs, ys, ps)

                gt = test['annotations'][i]['keypoints']
                for j, (x, y) in enumerate(zip(xs, ys)):
                    gtx = gt[3 * j]
                    gty = gt[3 * j + 1]
                    err = math.sqrt((gtx - x) ** 2 + (gty - y) ** 2)
                    if j == 0 or j >= 9:
                        continue
                    avg_err += err

                    rfx = coord_2D_rf[j][0]
                    rfy = coord_2D_rf[j][1]
                    err_rf = math.sqrt((gtx - rfx) ** 2 + (gty - rfy) ** 2)
                    avg_rf_err += err_rf

                    count += 1
                    cv2.circle(frame, (round(gtx), round(gty)), 1, (0, 255, 0), -1)
                    cv2.circle(frame, (round(x), round(y)), 1, (255, 0, 0), -1)
                    cv2.circle(frame, (round(rfx), round(rfy)), 1, (0, 0, 255), -1)
                    print(err, err_rf)
                    # if err_rf > 3:
                    #     cv2.circle(frame, (round(gtx), round(gty)), 1, (0, 255, 0), -1)
                    #     cv2.circle(frame, (round(x), round(y)), 1, (255, 0, 0), -1)
                    #     cv2.circle(frame, (round(rfx), round(rfy)), 1, (0, 0, 255), -1)
                    #     print(err, err_rf)

            cv2.imshow('point', frame)
            cv2.waitKey()


    print(avg_err / count, count)
    print(avg_rf_err / count, count)



def FPS_test():
    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    while True:
        if Tis.Snap_image(1) is True:
            start = time.time()
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():


                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                ps = [2, 4, 8]
                ds = [40 / lazer, 60 / lazer]
                tip_offset = 3.2
                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):
                    coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, x, y, ps)

                    tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], ds[0],
                                                        ds[1], mtx, tip_offset)
            end = time.time()
            cv2.imshow('Window', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()



def bending_odr_std():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    mtx, dist = camera_para_retrieve()

    length = 30
    loss_win = deque(maxlen=length)
    std_win = deque(maxlen=length)

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():
                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                m, b = line_polyfit(x, y)
                dx = x[1] - x[4]
                dy = y[1] - y[4]
                odr_end_pt = 9
                coord_2D_rf = []

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):
                    for i in range(1, odr_end_pt):
                        cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 0, 255), -1)
                        cv2.putText(frame, str(i), (round(kp[0][i][0]), round(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 0, 255), 1, cv2.LINE_AA)
                        kernel = kernel_choice(m, i, dx, dy)
                        rf_x, rf_y = edge_refinement_conv(gray_frame, x[i], y[i], kernel)
                        coord_2D_rf.append(np.array([rf_x, rf_y]))
                        cv2.circle(frame, (rf_x, rf_y), 1, (0, 255, 0), -1)

                    if coord_2D_rf:
                        interval = []
                        for i in range(1, 8):
                            dis = np.linalg.norm(coord_2D_rf[i] - coord_2D_rf[i - 1])
                            interval.append(dis)
                        interval[1] /= 3
                        interval[4] /= 3

                        dis_std = np.std(interval)
                        print(dis_std)
                        std_win.append(dis_std)

                        odr_model = odr.Model(target_function)
                        data = odr.Data(x[1:odr_end_pt], y[1:odr_end_pt])
                        ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0., 1.])
                        out = ordinal_distance_reg.run()
                        loss = out.sum_square

                        loss_win.append(loss)
                        # print(f"{mean(std_win):.2f}, {mean(loss_win):.2f}")
                        # print(f"{len(loss_win)}: {dis_std:.2f}, {loss:.2f}")
                        cv2.putText(frame, f'dist_std : {mean(std_win):.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f'loss : {mean(loss_win):.2f}', (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

                        # cv2.putText(frame, f'dist_std : {dis_std:.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        #             (0, 0, 255), 1, cv2.LINE_AA)
                        # cv2.putText(frame, f'loss : {loss:.2f}', (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                        #     1, cv2.LINE_AA)

                        if len(loss_win) == length:
                            print(f"Final {mean(std_win):.2f}, {mean(loss_win):.2f}")

            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                loss_win = deque(maxlen=length)
                std_win = deque(maxlen=length)
                print('Reset')


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()



def bending_odr_partial():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    mtx, dist = camera_para_retrieve()
    length = 15
    loss_win = deque(maxlen=length)

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

                seq_x, seq_y, seq = partial_filter(x, y)
                if len(seq) >= 4:

                    odr_model = odr.Model(target_function)
                    data = odr.Data(seq_x, seq_y)
                    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0, 1])
                    out = ordinal_distance_reg.run()
                    m_, b_ = out.beta
                    manual_loss = 0

                    for k in range(len(seq)):
                        cv2.circle(frame, (round(seq_x[k]), round(seq_y[k])), 1, (0, 255, 0), -1)
                        cv2.putText(frame, str(seq[k]), (round(seq_x[k]), round(seq_y[k])), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 1, cv2.LINE_AA)

                        dis = np.absolute(m_ * seq_x[k] - seq_y[k] + b_) / np.sqrt(m_ * m_ + 1)
                        manual_loss += dis
                    manual_loss /= len(seq)

                    cv2.putText(frame, f'{manual_loss:.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    loss_win.append(manual_loss)
                    p1 = (0, round(b_))
                    p2 = (1000, round(m_ * 1000 + b_))
                    cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

            if len(loss_win) == length:
                print(f"Final: {mean(loss_win):.4f}")
                loss_win = deque(maxlen=length)

            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                loss_win = deque(maxlen=length)
                print('Reset')


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def target_function(p, x):
    m, b = p
    return m * x + b

def detection_rate():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    detected_count = 0
    whole_count = 0
    initial = True
    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)
            # if diamondCorners:
            #     if initial:
            #         initial = False
            #         print(f'{tvec[0][0][2]:.2f}')
            #     cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)
            #     detected_count += 1



            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():
                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                seq_x, seq_y, seq, beta = partial_filter(x, y)
                if len(seq) >= 4:

                    plist, dlist, offset = pdlist_choosen_multi(seq)
                    if plist and dlist:
                        refine_plist = plist[0] + plist[1]
                        coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, x, y, refine_plist)
                        tip_rf, end_rf, ext_tip, ext_end = scale_estimation_multi_mod(coord_3D_rf[0],
                                                                                      coord_3D_rf[1],
                                                                                      coord_3D_rf[2], dlist[0][0],
                                                                                      dlist[0][1], mtx, offset[0])
                        tip_rf2, end_rf2, ext_tip2, ext_end2 = scale_estimation_multi_mod(coord_3D_rf[3],
                                                                                          coord_3D_rf[4],
                                                                                          coord_3D_rf[5],
                                                                                          dlist[1][0],
                                                                                          dlist[1][1], mtx,
                                                                                          offset[1])
                        tip_diff = np.linalg.norm(tip_rf - tip_rf2)

                        if tip_diff < 1 + 0.5 * (8 - len(seq)):
                            if initial:
                                initial = False
                                print(f'{tip_rf}')
                            detected_count += 1

                            # draw
                            endpts = line_define(seq_x, beta)
                            cv2.line(frame, endpts[0], endpts[1], (0, 0, 255), 2, cv2.LINE_AA)
                            for k in range(len(coord_2D_rf)):
                                cv2.circle(frame, (coord_2D_rf[k][0], coord_2D_rf[k][1]), 5, (0, 255, 0), -1)
                                cv2.putText(frame, str(refine_plist[k]), (coord_2D_rf[k][0], coord_2D_rf[k][1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

            whole_count += 1

            if whole_count >= 100:
                print(f'{detected_count / whole_count:.2f}')
                detected_count = 0
                whole_count = 0
                initial = True

            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)

            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                detected_count = 0
                whole_count = 0
                initial = True
                print('Reset')

    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # model_keypoint_error()
    # bending_odr_partial()
    # bending_odr_std()
    # aruco_accuracy_record()
    # multi_set_needle_refinement_accuracy_record()
    # needle_refinement_accuracy_record()
    # FPS_test()
    # bending_odr_std()
    detection_rate()