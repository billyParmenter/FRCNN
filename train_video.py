import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode

# Register your custom dataset
register_coco_instances("your_dataset_train", {}, "./data/frc/train/_annotations.coco.json", "./data/frc/train")
register_coco_instances("your_dataset_test", {}, "./data/frc/test/_annotations.coco.json", "./data/frc/test")

# Load a Faster R-CNN configuration
cfg = get_cfg()
cfg.merge_from_file("/home/billy/src/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("your_dataset_train",)
cfg.DATASETS.TEST = ("your_dataset_test",)  # Use your test dataset here
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 4000
cfg.SOLVER.BASE_LR = 0.001
cfg.OUTPUT_DIR = "output_directory"  # Directory to save checkpoints and logs

# Create output directory if it doesn't exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
