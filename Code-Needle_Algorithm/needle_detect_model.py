
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

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


cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 11
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((11, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)


import time
import sys
import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst


def main():
    import sys
    # sys.path.append("../python-common")
    import cv2
    import TIS
    mtx, dist = camera_para_retrieve()

    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)

    Tis.Start_pipeline()

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            frame = np.array(frame)
            outputs = predictor(frame)
            diamondCorners, rvec, tvec = diamond_detection(frame, mtx, dist)
            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()
            if kp.shape[0] != 0:
                for i in range(10):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)



                coord_3D = []
                for i in range(3):
                    pt = np.array([kp[0][i][0], kp[0][i][1], 0], dtype='float64')
                    coord_3D.append(pt)

                est_tvec = scale_estimation(coord_3D[2], coord_3D[1], coord_3D[0], 32, 30, mtx, dist)
                # est_tvec_rev = scale_estimation(coord_3D[0], coord_3D[1], coord_3D[2], 15, 32, mtx, dist)
                if diamondCorners != None:
                    print('gt', tvec)
                    print('est', est_tvec)
                    # print('est2', est_tvec_rev)
                    print("------------------")
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            cv2.imshow('Window', frame)
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()
    print('Program ends')

if __name__ == "__main__":
    main()
