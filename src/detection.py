import copy
import random

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetMapper, MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

import torch, cv2, os
import numpy as np
import datetime


IMAGE_PATH = "zdjecia/puzzle2.jpg"
OUTPUT_DIR = "pieces"


class PuzzleDetector:

    cfg: detectron2.config.CfgNode
    model_path: str

    def register_database(self):
        register_coco_instances("puzzle_train", {}, "puzzle_database/puzzle.json", "puzzle_database")
    

    def create_config(self):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE='cuda'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("puzzle_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = 0.00033  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.RANDOM_FLIP = "horizontal"


    def data_augmentation(self):
        augs = [
        T.RandomBrightness(0.8, 1.2),
        T.RandomFlip(prob=0.5),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        #T.RandomCrop("absolute", (640, 640))
        ]
        return augs

    def train(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.model_path = os.path.join(self.cfg.OUTPUT_DIR, str(now))
        os.makedirs(self.model_path, exist_ok=True)
        self.cfg.OUTPUT_DIR = self.model_path
        trainer = DefaultTrainer(self.cfg) 
        dataloader = build_detection_train_loader(self.cfg,
            mapper=DatasetMapper(self.cfg, is_train=True, augmentations=
                self.data_augmentation())
        )
        trainer.build_train_loader = dataloader
        trainer.resume_or_load(resume=False)
        trainer.train()


    def predict(self):
        #self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        self.cfg.MODEL.WEIGHTS = "./output\\2026-05-14-14-21-49\model_final.pth"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        predictor = DefaultPredictor(self.cfg)
        puzzle_metadata = MetadataCatalog.get("puzzle_train")
        img = cv2.imread(IMAGE_PATH)
        outputs = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=puzzle_metadata, scale=0.5)
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Predicted puzzles", out.get_image()[:, :, ::-1])
        cv2.waitKey()
