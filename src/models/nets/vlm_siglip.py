import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from .base import BaseModel

class VLM_SigLIP(BaseModel):
    __tag_name_huggingface__ = "google/siglip-base-patch16-224"

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__(num_classes, freeze_backbone)
        
        self.processor = AutoProcessor.from_pretrained(self.__tag_name_huggingface__)
        self.backbone = AutoModel.from_pretrained(self.__tag_name_huggingface__)
        self.freeze()
        
        hidden_size = self.backbone.config.vision_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        features = self.backbone.get_image_features(pixel_values)
        
        if self.classifier.weight.device != features.device:
            self.classifier = self.classifier.to(features.device)
            
        return self.classifier(features)