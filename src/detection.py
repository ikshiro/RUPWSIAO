import random

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import torch, cv2, os
import numpy as np


IMAGE_PATH = "zdjecia/kilka.jpg"
OUTPUT_DIR = "pieces"


class PuzzleDetector:

    cfg: detectron2.CfgNode


    def register_database():
        register_coco_instances("puzzle_train", {}, "puzzle_database/puzzle.json", "puzzle_database")
    

    def create_config(self):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE='cpu'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("puzzle_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()


    def predict(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        predictor = DefaultPredictor(self.cfg)
        puzzle_metadata = MetadataCatalog.get("puzzle_train")
        img = cv2.imread(IMAGE_PATH)
        outputs = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=puzzle_metadata, scale=0.5)
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Predicted puzzles", out.get_image()[:, :, ::-1])
        cv2.waitKey()
