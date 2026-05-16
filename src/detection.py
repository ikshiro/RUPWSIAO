import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
import cv2, os
import datetime


IMAGE_PATH = "zdjecia/puzzle.jpg"
DATASET_NAME = "puzzle_train"


class PuzzleDetector:

    cfg: detectron2.config.CfgNode
    model_path: str


    def __init__(self):
        self._register_database()
        self._create_config()


    def _register_database(self):
        register_coco_instances(DATASET_NAME, {}, "puzzle_database/puzzle.json", "puzzle_database")
    

    def _create_config(self):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE='cpu'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = (DATASET_NAME,)
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


    def _data_augmentation(self):
        augs = [
        T.RandomBrightness(0.8, 1.2),
        T.RandomFlip(prob=0.5),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ]
        return augs
    

    def _show_predictions(self, outputs, img):
        puzzle_metadata = MetadataCatalog.get(DATASET_NAME)
        visualizer = Visualizer(img[:, :, ::-1], metadata=puzzle_metadata, scale=0.5)
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Predicted puzzles", out.get_image()[:, :, ::-1])
        cv2.waitKey()


    def predict(self, show_predictions = False):
        #self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        self.cfg.MODEL.WEIGHTS = "./output/2026-05-14-14-21-49/model_final.pth"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        predictor = DefaultPredictor(self.cfg)
        img = cv2.imread(IMAGE_PATH)
        outputs = predictor(img)
        if show_predictions:
            self._show_predictions(outputs, img)
        contours = []
        for pred_mask in outputs['instances'].pred_masks:
            mask = pred_mask.numpy().astype('uint8')
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours.append(contour[0][0])
        return contours
    

    def train(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.model_path = os.path.join(self.cfg.OUTPUT_DIR, str(now))
        os.makedirs(self.model_path, exist_ok=True)
        self.cfg.OUTPUT_DIR = self.model_path
        trainer = DefaultTrainer(self.cfg) 
        dataloader = build_detection_train_loader(self.cfg,
            mapper=DatasetMapper(self.cfg, is_train=True, augmentations=
                self._data_augmentation())
        )
        trainer.build_train_loader = dataloader
        trainer.resume_or_load(resume=False)
        trainer.train()

