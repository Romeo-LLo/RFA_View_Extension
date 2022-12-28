import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random, math
import torch
import glob
import matplotlib.pyplot as plt
from needle_utils import *
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import warnings
warnings.filterwarnings("ignore")

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    frameS = cv2.resize(im, (1080, 810))
    cv2.imshow('Window', frameS)
    cv2.waitKey()


keypoint_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
keypoint_flip_map = []

MetadataCatalog.get("needle_train").thing_classes = ["edge"]
MetadataCatalog.get("needle_train").thing_dataset_id_to_contiguous_id = {1: 0}
MetadataCatalog.get("needle_train").keypoint_names = keypoint_names
MetadataCatalog.get("needle_train").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("needle_train").evaluator_type = "coco"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "../Model_path/model_1227.pth"  # path to the model we trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)
root = "../All_images/TrainImg_1222_labeled"


def test_folder():
    images = glob.glob(f'../All_images/TestImg1222/*.jpg')

    for img in images:
        frame = cv2.imread(img)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():
            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]
            seq_x, seq_y = partial_filter(x, y)
            for k in range(len(seq_x)):
                cv2.circle(frame, (round(seq_x[k]), round(seq_y[k])), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(k), (round(seq_x[k]), round(seq_y[k])), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 1, cv2.LINE_AA)

        frameS = cv2.resize(frame, (720, 540))
        cv2.imshow('win', frameS)
        cv2.waitKey()



def partial_filter(x, y):
    valid_seq = []
    cand_seq = []
    cursor = 1
    while cursor < 7:
        x_cont = x[cursor: cursor+4]
        y_cont = y[cursor: cursor+4]
        if isMonotonic(x_cont) and isMonotonic(y_cont) and isDistinct(x_cont, y_cont):
            if not valid_seq:
                valid_seq.append(cursor)
                valid_seq.append(cursor+1)
                valid_seq.append(cursor+2)
                valid_seq.append(cursor+3)

            else:
                valid_seq.append(cursor+3)
        else:
            if valid_seq:
                cand_seq.append(valid_seq)
            valid_seq = []
        cursor += 1
    cand_seq.append(valid_seq)

    cand_len = [len(cand) for cand in cand_seq]
    f = lambda i: cand_len[i]
    max_arg = max(range(len(cand_len)), key=f)

    seq = cand_seq[max_arg]
    seq_x = x[seq]
    seq_y = y[seq]
    return seq_x, seq_y, seq


if __name__ == "__main__":
    test_folder()