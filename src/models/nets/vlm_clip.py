import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from .base import BaseModel

class VLM_CLIP(BaseModel):
    __tag_name_huggingface__ = "openai/clip-vit-base-patch16"

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__(num_classes, freeze_backbone)
        
        self.processor = CLIPProcessor.from_pretrained(self.__tag_name_huggingface__)
        self.backbone = CLIPModel.from_pretrained(self.__tag_name_huggingface__)
        self.freeze()
        
        # Cria a cabeça de classificação baseada na dimensão do CLIP
        hidden_size = self.backbone.vision_model.config.projection_dim
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        # Em modelos multimodais base, pegamos as features da imagem e passamos na nossa Linear Probe
        features = self.backbone.get_image_features(pixel_values)
        
        # Garante que o classifier está no mesmo device da GPU
        if self.classifier.weight.device != features.device:
            self.classifier = self.classifier.to(features.device)
            
        return self.classifier(features)