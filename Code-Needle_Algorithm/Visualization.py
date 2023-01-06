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
from partial_filter import *
import torch
import gi
import cv2.aruco as aruco
import datetime
import time
from scipy.linalg import norm
from collections import deque
from statistics import mean, stdev

mtx, dist = camera_para_retrieve()

gi.require_version("Gst", "1.0")
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_1227.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)

num_steps = 2
num_lines = 2


def setup_plot():
    plt.ion()
    fig = plt.figure(figsize=(5, 10))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.65,
                        hspace=0.4)
    ax3D = fig.add_subplot(211, projection='3d')
    ax2D = fig.add_subplot(212, aspect='equal')


    color = ['red', 'green']
    line3D = ax3D.plot([], [], [], color[0], linewidth=1)[0]
    line2D = [ax2D.plot([], [], color[0], linewidth=1)[0] for i in range(num_lines)]

    sq_len = 15
    shallow = 0
    deep = 65
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')
    ax3D.set_xlim(-sq_len-4, sq_len-4)
    ax3D.set_ylim(deep, shallow)
    ax3D.set_zlim(-sq_len, sq_len)
    ax3D.set_xticklabels([])
    ax3D.set_yticklabels([])
    ax3D.set_zticklabels([])

    us_zone = 17
    us_x, us_z = np.meshgrid(range(-us_zone-4, us_zone-4), range(-us_zone, us_zone))
    us_y = np.full((2 * us_zone, 2 * us_zone), shallow)
    ax3D.plot_surface(us_x, us_y, us_z, alpha=0.5)

    sq_len = 30
    ax2D.set_xlabel('X')
    ax2D.set_ylabel('Y')
    ax2D.set_xlim(-sq_len, sq_len)
    ax2D.set_ylim(-sq_len, sq_len)
    ax2D.set_yticklabels([])
    ax2D.set_xticklabels([])


    us_img = plt.imread('../All_images/ultrasound_show.png')
    ax2D.imshow(us_img, extent=[-sq_len, sq_len, -sq_len, sq_len])

    return fig, plt, ax3D, ax2D, line3D, line2D



def realtime_hybrid_visual():

    fig, plt, ax3D, ax2D, line3D, line2D = setup_plot()
    sm_factor = 0.6
    first = True

    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()


    while True:
        if Tis.Snap_image(1) is True:

            traj3D = np.zeros((num_steps, 3))
            traj2D = np.zeros((num_steps, 2))

            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)
            if diamondCorners:
                tip_a, end_a = pose_trans_needle(tvec, rvec)
                traj3D = np.array([tip_a, end_a])
                traj2D = np.array([tip_a[:2], end_a[:2]])
            else:
                outputs = predictor(frame)
                kp_tensor = outputs["instances"].pred_keypoints

                if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():
                    kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                    x = kp[0, :-1, 0]
                    y = kp[0, :-1, 1]
                    if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):
                        m, b = line_polyfit(x, y)
                        p1 = (0, round(b))
                        p2 = (1500, round(m * 1500 + b))
                        cv2.line(frame, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

                        for i in range(11):
                            cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 5, (0, 255, 0), -1)

                        coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, x, y, plists[0])

                        tip_rf, end_rf = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlists[0][0],
                                                          dlists[0][1], mtx, tip_offset[0])

                        traj3D = np.array([tip_rf, end_rf])
                        traj2D = np.array([tip_rf[:2], end_rf[:2]])

            # if first:
            #     pre_traj3D = traj3D
            #     first = False
            # else:
            #     sm = pre_traj3D * sm_factor + traj3D * (1 - sm_factor)
            #     traj3D = np.array([sm[0], sm[1]])
            #     traj2D = np.array([sm[0][:2], sm[1][:2]])

            line3D.set_data(traj3D[:, 0], traj3D[:, 2])
            line3D.set_3d_properties(-traj3D[:, 1])
            line2D.set_data(-traj2D[:, 0], -traj2D[:, 1])

            fig.canvas.draw()
            fig.canvas.flush_events()
            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)

            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                plt.savefig("../All_images/demo.jpg")
                cv2.imwrite("../All_images/demo2.jpg", frame)
                print('Record')

    Tis.Stop_pipeline()
    cv2.destroyAllWindows()

def realtime_hybrid_visual_partial():

    fig, plt, ax3D, ax2D, line3D, line2D = setup_plot()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    while True:
        if Tis.Snap_image(1) is True:

            traj3D = np.zeros((num_steps, 3))
            traj2D = np.zeros((num_lines, num_steps, 2))
            linewidth = 1

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
                seq_x, seq_y, seq, beta = partial_filter(x, y)
                if len(seq) >= 4:

                    plist, dlist, offset = pdlist_choosen(seq)
                    if plist and dlist:
                        coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, x, y, plist)
                        tip_rf, end_rf, ext_tip, ext_end = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0],
                                                          dlist[1], mtx, offset)


                        traj3D = np.array([ext_tip, ext_end])
                        linewidth = 9 - len(seq)
                        tip_up2D, tip_low2D, end_up2D, end_low2D = parallel_line_pts(ext_tip, ext_end, linewidth)
                        traj2D[0] = np.array([tip_up2D, end_up2D])
                        traj2D[1] = np.array([tip_low2D, end_low2D])
                        # draw
                        for k in range(len(seq)):
                            cv2.circle(frame, (round(seq_x[k]), round(seq_y[k])), 3, (0, 0, 255), -1)
                            cv2.putText(frame, str(seq[k]), (round(seq_x[k]), round(seq_y[k])),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 1, cv2.LINE_AA)
                        endpts = line_define(seq_x, beta)
                        cv2.line(frame, endpts[0], endpts[1], (0, 0, 255), 2, cv2.LINE_AA)

            ax3D.lines[0].set_linewidth(linewidth)

            line3D.set_data(traj3D[:, 0], traj3D[:, 2])
            line3D.set_3d_properties(-traj3D[:, 1])
            line2D[0].set_data(-traj2D[0][:, 0], -traj2D[0][:, 1])
            line2D[1].set_data(-traj2D[1][:, 0], -traj2D[1][:, 1])


            fig.canvas.draw()
            fig.canvas.flush_events()
            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                plt.savefig("../All_images/demo.jpg")
                cv2.imwrite("../All_images/demo2.jpg", frame)
                print('Record')


    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def realtime_hybrid_visual_partial_multi():
    fig, plt, ax3D, ax2D, line3D, line2D = setup_plot()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()
    length = 10
    odr_win = deque(maxlen=length)
    while True:
        if Tis.Snap_image(1) is True:

            traj3D = np.zeros((num_steps, 3))
            traj2D = np.zeros((num_lines, num_steps, 2))
            linewidth = 1

            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)

            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)
            if diamondCorners:
                cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)
                tip_a, end_a = pose_trans_needle(tvec, rvec)
                traj3D = np.array([tip_a, end_a])
                tip_up2D, tip_low2D, end_up2D, end_low2D = parallel_line_pts(tip_a, end_a, linewidth)
                traj2D[0] = np.array([tip_up2D, end_up2D])
                traj2D[1] = np.array([tip_low2D, end_low2D])

            else:
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
                            if coord_2D_rf and coord_2D_rf:
                                tip_rf, end_rf, ext_tip, ext_end = scale_estimation_multi_mod(coord_3D_rf[0], coord_3D_rf[1], coord_3D_rf[2], dlist[0][0],
                                                                                              dlist[0][1], mtx, offset[0])
                                tip_rf2, end_rf2, ext_tip2, ext_end2 = scale_estimation_multi_mod(coord_3D_rf[3], coord_3D_rf[4], coord_3D_rf[5], dlist[1][0],
                                                                  dlist[1][1], mtx, offset[1])
                                tip_diff = np.linalg.norm(tip_rf - tip_rf2)

                                if tip_diff < 1.5 + 0.5 * (8 - len(seq)):
                                    traj3D = np.array([ext_tip, ext_end])
                                    linewidth = 9 - len(seq)
                                    tip_up2D, tip_low2D, end_up2D, end_low2D = parallel_line_pts(ext_tip, ext_end, linewidth)
                                    traj2D[0] = np.array([tip_up2D, end_up2D])
                                    traj2D[1] = np.array([tip_low2D, end_low2D])

                                  # draw
                                    endpts = line_define(seq_x, beta)
                                    cv2.line(frame, endpts[0], endpts[1], (0, 0, 255), 2, cv2.LINE_AA)
                                    for k in range(len(coord_2D_rf)):
                                        cv2.circle(frame, (coord_2D_rf[k][0], coord_2D_rf[k][1]), 5, (0, 255, 0), -1)
                                        cv2.putText(frame, str(refine_plist[k]), (coord_2D_rf[k][0], coord_2D_rf[k][1]),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                                    # bending
                                    odr_model = odr.Model(target_function)
                                    data = odr.Data(seq_x, seq_y)
                                    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[beta[0], beta[1]])
                                    out = ordinal_distance_reg.run()
                                    dis = [np.sqrt(dx ** 2 + dy ** 2) for dx, dy in zip(out.delta, out.eps)]
                                    odr_err = sum(dis) / len(seq)
                                    odr_win.append(odr_err)

                                    if mean(odr_win) > 1:
                                        cv2.putText(frame, 'Bending detected!', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                                    1, (0, 0, 255), 1, cv2.LINE_AA)
                                    else:
                                            cv2.putText(frame, f'Normal', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                                    1, (0, 255, 0), 1, cv2.LINE_AA)
                                        # cv2.putText(frame, f'{odr_err:.2f}', (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                        #             1, (0, 255, 0), 1, cv2.LINE_AA)
                                # else:
                                #     print(seq, tip_diff)

            ax3D.lines[0].set_linewidth(linewidth)
            line3D.set_data(traj3D[:, 0], traj3D[:, 2])
            line3D.set_3d_properties(-traj3D[:, 1])
            line2D[0].set_data(-traj2D[0][:, 0], -traj2D[0][:, 1])
            line2D[1].set_data(-traj2D[1][:, 0], -traj2D[1][:, 1])

            fig.canvas.draw()
            fig.canvas.flush_events()
            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)


            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                odr_win = deque(maxlen=length)

                # plt.savefig("../All_images/demo.jpg")
                # cv2.imwrite("../All_images/demo2.jpg", frame)
                print('Reset')

    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # realtime_hybrid_visual()
    # realtime_hybrid_visual_partial()
    # predicted_zone(1, 2)
    realtime_hybrid_visual_partial_multi()

