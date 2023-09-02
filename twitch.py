import utils
import config

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load your pre-trained PyTorch object detection model
device = torch.device("cpu")
saved_name = './result/small.pth'
checkpoint = torch.load(saved_name, map_location=device)
model = utils.get_model_object_detector(config.num_classes)  # Replace with your model creation code
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# Define the function for object detection
def detect_objects(frame):
    # Preprocess the frame
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    input_tensor = transform(frame).unsqueeze(0)

    with torch.no_grad():
        detections = model(input_tensor)

    # Process detection results as needed
    for score, label, box in zip(detections[0]['scores'], detections[0]['labels'], detections[0]['boxes']):
        if score > config.detection_threshold:  # Adjust the confidence threshold as needed
            xmin, ymin, xmax, ymax = box.tolist()
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return frame

# Access the Twitch livestream using OpenCV VideoCapture
# Replace 'YOUR_STREAM_URL' with the actual Twitch stream URL
stream_url = 'https://www.twitch.tv/firstcanada'

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open the stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        processed_frame = detect_objects(frame)

        # Display the processed frame
        cv2.imshow('Livestream with Object Detection', processed_frame)

        # Press 'q' to exit the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
