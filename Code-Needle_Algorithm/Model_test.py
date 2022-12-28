import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random, math
import torch
import glob
import matplotlib.pyplot as plt

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
cfg.MODEL.WEIGHTS = "../Model_path/model_final.pth"  # path to the model we trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)
root = "../All_images/TrainImg_1222_labeled"

def test_json():
    avg_err = 0
    count = 0

    f = open("../All_images/TrainImg_1222_labeled/train.json", newline='')
    test = json.load(f)
    imgs = test['images']
    for i in range(len(imgs)):
        print(imgs[i]['file_name'])
        img_path = os.path.join(root, imgs[i]['file_name'])
        frame = cv2.imread(img_path)

        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints

        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():
            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score

            zero_pos = 900
            for k in range(11):
                cv2.circle(frame, (int(kp[0][k][0]), int(kp[0][k][1])), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(k), (round(kp[0][k][0]), round(kp[0][k][1])), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'{k}: {kp[0][k][2]:.2f}', (100, zero_pos-50*k), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 0, 0), 1, cv2.LINE_AA)
            xs = kp[0, :-1, 0]
            ys = kp[0, :-1, 1]

            gt = test['annotations'][i]['keypoints']
            for j, (x, y) in enumerate(zip(xs, ys)):
                gtx = gt[3 * j]
                gty = gt[3 * j + 1]
                err = math.sqrt((gtx - x) ** 2 + (gty - y) ** 2)
                if j == 0 or j >= 9:
                    continue
                avg_err += err
                count += 1
                # print(err)

        frameS = cv2.resize(frame, (720, 540))
        cv2.imshow('win', frameS)
        cv2.waitKey()
        # cv2.imwrite(f'../All_images/TrainImg_1222_labeled/{i}.png', frame)

    print(avg_err / count, count)


def test_folder():
    images = glob.glob(f'../All_images/TestImg1222/*.jpg')

    for img in images:
        frame = cv2.imread(img)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():
            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            zero_pos = 900
            for k in range(11):
                cv2.circle(frame, (int(kp[0][k][0]), int(kp[0][k][1])), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(k), (round(kp[0][k][0]), round(kp[0][k][1])), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'{k}: {kp[0][k][2]:.2f}', (100, zero_pos - 50 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 0, 0), 1, cv2.LINE_AA)
        frameS = cv2.resize(frame, (720, 540))
        cv2.imshow('win', frameS)
        cv2.waitKey()

if __name__ == "__main__":
    test_folder()