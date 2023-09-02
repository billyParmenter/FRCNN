import cv2
import os
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load the configuration and model
cfg = get_cfg()
cfg.merge_from_file("../../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join("./output_directory/model_final.pth")  # Path to your trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for object detection
predictor = DefaultPredictor(cfg)

# Open webcam
cam = cv2.VideoCapture(0)  # 0 indicates the default camera, change to a different value if needed

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Perform inference on the frame
    outputs = predictor(frame)

    # Visualize the results
    v = Visualizer(frame[:, :, ::-1], scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_frame = v.get_image()[:, :, ::-1]

    # Display the frame with predictions
    cv2.imshow("Inference", result_frame)

    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()
