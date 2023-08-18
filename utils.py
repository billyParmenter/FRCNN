import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import json
import os

def register_datasets():
    # Specify the paths to your JSON annotation files and image root directories
    train_json_path = './data/frc/train/_annotations.coco.json'
    test_json_path = "./data/frc/test/_annotations.coco.json"
    train_image_root = "./data/frc/train"
    test_image_root = "./data/frc/test"

    # Register the train dataset
    DatasetCatalog.register("train", lambda: load_coco_json(train_json_path, train_image_root))
    MetadataCatalog.get("train").set(thing_classes=["cubes_cones", "cube", "cone", "tipped"])  # List your class names

    # Register the test dataset
    DatasetCatalog.register("test", lambda: load_coco_json(test_json_path, test_image_root))
    MetadataCatalog.get("test").set(thing_classes=["cubes_cones", "cube", "cone", "tipped"])  # List your class names

def load_coco_json(json_file, image_root):
    with open(json_file) as f:
        coco_json = json.load(f)
    dataset_dicts = []
    for idx, entry in enumerate(coco_json["images"]):
        record = {}
        record["file_name"] = os.path.join(image_root, entry["file_name"])
        record["image_id"] = idx
        record["height"] = entry["height"]
        record["width"] = entry["width"]
        ann_ids = coco_json.get("annotations", [])
        objs = []
        for ann in ann_ids:
            if ann["image_id"] == entry["id"]:
                obj = {
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": ann["category_id"],
                }
                objs.append(obj)
        record["annotations"] = objs
        
        # Loop through the annotations and assign a default category to images without categories
        default_category_id = -1  # Choose an appropriate default category ID
        for annotation in record["annotations"]:
            if "category_id" not in annotation:
                annotation["category_id"] = default_category_id
        
        dataset_dicts.append(record)
    return dataset_dicts



# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_object_detector(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def save_model(name, epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'result/{name}')
