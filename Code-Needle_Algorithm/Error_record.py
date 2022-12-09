
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
cfg.MODEL.WEIGHTS = '../Model_path/model_1201.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
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
                error = error_calc_board(tip_a, anchor)

                # error = error_angle_board(tip_a, end_a, anchor=anchor)

                if error < 1.5:
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

                        tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], ds[0],
                                                          ds[1], mtx, tip_offset[i])
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
                coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod(gray_frame, xs, ys, ps)

                gt = test['annotations'][i]['keypoints']
                for j, (x, y) in enumerate(zip(xs, ys)):
                    gtx = gt[3 * j]
                    gty = gt[3 * j + 1]
                    err = math.sqrt((gtx - x) ** 2 + (gty - y) ** 2)
                    if err > 100 or j == 0 or j >= 9:
                        continue
                    avg_err += err

                    rfx = coord_2D_rf[j][0]
                    rfy = coord_2D_rf[j][1]
                    err_rf = math.sqrt((gtx - rfx) ** 2 + (gty - rfy) ** 2)
                    avg_rf_err += err_rf

                    count += 1

                    if err_rf > 3:
                        cv2.circle(frame, (round(gtx), round(gty)), 1, (0, 255, 0), -1)
                        cv2.circle(frame, (round(x), round(y)), 1, (255, 0, 0), -1)
                        cv2.circle(frame, (round(rfx), round(rfy)), 1, (0, 0, 255), -1)
                        print(err, err_rf)

            cv2.imshow('point', frame)
            cv2.waitKey()

    print(avg_err / count, count)
    print(avg_rf_err / count, count)


if __name__ == "__main__":
    model_keypoint_error()
    # aruco_accuracy_record()
    # multi_set_needle_refinement_accuracy_record()
    # needle_refinement_accu       racy_record()
