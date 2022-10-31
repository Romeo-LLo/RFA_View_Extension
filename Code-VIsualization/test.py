import numpy as np
import matplotlib.pyplot as plt
from needle_utils import *
from Linear_equation import *
import TIS
import cv2
import torch
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
setup_logger()

cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'model_final3.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)

def visual():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line = ax.plot([], [], [])[0]
    sq_len = 10
    deep = 80
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-sq_len, sq_len)
    ax.set_ylim(40, deep)
    ax.set_zlim(-sq_len, sq_len)

    mtx, dist = camera_para_retrieve()
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)


            diamondCorners, rvec, tvec = diamond_detection(dis_frame, mtx, dist)


            if diamondCorners:

                p1t = pose_trans_needle(tvec, rvec, 18)
                p4t = pose_trans_needle(tvec, rvec, 3)

                num_steps = 15
                traj = np.linspace(p1t, p4t, num=num_steps)
                print(traj)
                line.set_data(traj[:, 0], traj[:, 2])
                line.set_3d_properties(-traj[:, 1])
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()

            frameS = cv2.resize(frame, (900, 675))

            cv2.imshow('Window', frameS)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    visual()
