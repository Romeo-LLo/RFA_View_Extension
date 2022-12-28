import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T








class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomFlip(prob=0.5),
            T.RandomRotation((-20, 20))
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('vis', im)
    cv2.waitKey()
    plt.figure()
    plt.imshow(im)
    plt.axis('off')




register_coco_instances("needle_train", {}, "../All_images/TrainImg_1222_labeled/train.json", "../All_images/TrainImg_1222_labeled/")
register_coco_instances("needle_test", {}, "../All_images/TrainImg_1222_labeled/test.json", "../All_images/TrainImg_1222_labeled/")
keypoint_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
keypoint_flip_map = []

MetadataCatalog.get("needle_train").thing_classes = ["edge"]
MetadataCatalog.get("needle_train").thing_dataset_id_to_contiguous_id = {1:0}
MetadataCatalog.get("needle_train").keypoint_names = keypoint_names
MetadataCatalog.get("needle_train").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("needle_train").evaluator_type="coco"

dataset_dicts = DatasetCatalog.get("needle_train")
needle_metadata = MetadataCatalog.get("needle_train")

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=needle_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2_imshow(vis.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("needle_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "../Model_path/model_1225.pth"  # path to the model we trained
cfg.OUTPUT_DIR = "../Model_path"

cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []         # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()



trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()