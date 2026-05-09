import random

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import torch, cv2, os
import numpy as np


IMAGE_PATH = "zdjecia/kilka.jpg"
OUTPUT_DIR = "pieces"

image = cv2.imread(IMAGE_PATH)
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'

register_coco_instances("puzzle_train", {}, "puzzle_database/puzzle.json", "puzzle_database")
dataset_dicts = DatasetCatalog.get("puzzle_train")
balloon_metadata = MetadataCatalog.get("puzzle_train")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("aaa", out.get_image()[:, :, ::-1])
    cv2.waitKey()
