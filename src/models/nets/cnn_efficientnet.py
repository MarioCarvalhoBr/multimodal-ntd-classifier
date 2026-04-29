from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from .base import BaseModel

class CNN_EfficientNetB3(BaseModel):
    __tag_name_huggingface__ = "google/efficientnet-b3"

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__(num_classes, freeze_backbone)
        
        self.processor = EfficientNetImageProcessor.from_pretrained(self.__tag_name_huggingface__)
        
        self.backbone = EfficientNetForImageClassification.from_pretrained(
            self.__tag_name_huggingface__,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.freeze()

    def freeze(self):
        if self.freeze_backbone:
            for param in self.backbone.efficientnet.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        return self.backbone(pixel_values).logits