import torch, detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt

import time
from detectron2.data.datasets import register_coco_instances
register_coco_instances("needle_test", {}, "../Needle keypoints/test.json", "../Needle keypoints")


keypoint_names = ['a', 'b', 'c', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k']
keypoint_flip_map = []
krule = [('a', 'b', (255, 0 , 0)), ('b', 'c', (255, 127, 0	)), ('c', 'd', (255, 255, 0	)), ('d', 'e', (0, 255, 0)), ('e', 'f', (0, 0, 255	)),
         ('f', 'g', (75, 0, 130	)), ('g', 'h', (148, 0, 211	)), ('h', 'i', (210,105,30)), ('i', 'j', (176,196,222))]
MetadataCatalog.get("needle_test").thing_classes = ["labels_all"]
MetadataCatalog.get("needle_test").thing_dataset_id_to_contiguous_id = {1:0}
MetadataCatalog.get("needle_test").keypoint_names = keypoint_names
MetadataCatalog.get("needle_test").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("needle_test").evaluator_type="coco"
MetadataCatalog.get("needle_test").keypoint_connection_rules = krule






cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
# load the pre trained model from Detectron2 model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
# set confidence threshold for this model

# load model weights
cfg.MODEL.WEIGHTS = 'model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
# create the predictor for pose estimation using the config
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 11
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((11, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)



test_imgs = os.listdir("../All_image")
for i in range(len(test_imgs)):

    start = time.time()
    im = cv2.imread(os.path.join("../All_image", test_imgs[i]))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # keypoints = outputs["instances"].pred_keypoints
    # for i in range(10):
    #     cv2.circle(im, (keypoints[0][i][]))
    v = Visualizer(im[:,:,::-1], MetadataCatalog.get("none"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('test', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

    # end = time.time()
    # period = end - start
    # print(1 / period)
