import torch.nn as nn
import torchvision.models as models


class DynamicBoxPredictionNet(nn.Module):
    def __init__(self, num_classes):
        super(DynamicBoxPredictionNet, self).__init__()
        
        # Load a pre-trained CNN backbone (e.g., ResNet, VGG)
        self.cnn_backbone = models.resnet50(pretrained=True)
        
        # Remove the fully connected layers from the backbone
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])
        
        # Additional layers for dynamic box prediction
        self.num_classes = num_classes
        
        # Fully connected layers for class prediction
        self.class_fc = nn.Linear(2048, num_classes)
        
        # Fully connected layers for bounding box prediction
        self.box_fc = nn.Linear(2048, num_classes * 4)  # Each box has 4 parameters
        
        # Fully connected layer for box count prediction
        self.box_count_fc = nn.Linear(2048, 1)  # Predict the number of boxes
        
    def forward(self, x):
        # Feature extraction using the CNN backbone
        features = self.cnn_backbone(x)
        features = features.view(features.size(0), -1)
        
        # Class prediction
        class_preds = self.class_fc(features)
        
        # Bounding box prediction
        box_preds = self.box_fc(features)
        
        # Box count prediction
        box_count_pred = self.box_count_fc(features)
        
        return class_preds, box_preds, box_count_pred