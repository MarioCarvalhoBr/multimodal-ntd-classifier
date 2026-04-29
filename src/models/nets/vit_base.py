from transformers import ViTImageProcessor, ViTForImageClassification
from .base import BaseModel

class ViT_BasePatch16_224(BaseModel):
    __tag_name_huggingface__ = "google/vit-base-patch16-224"

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__(num_classes, freeze_backbone)
        
        self.processor = ViTImageProcessor.from_pretrained(self.__tag_name_huggingface__)
        
        self.backbone = ViTForImageClassification.from_pretrained(
            self.__tag_name_huggingface__,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        self.freeze()

    def freeze(self):
        if self.freeze_backbone:
            for param in self.backbone.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        return self.backbone(pixel_values).logits