from transformers import AutoImageProcessor, ResNetForImageClassification
from .base import BaseModel

class CNN_ResNet50(BaseModel):
    __tag_name_huggingface__ = "microsoft/resnet-50"

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__(num_classes, freeze_backbone)
        
        self.processor = AutoImageProcessor.from_pretrained(self.__tag_name_huggingface__)
        
        # ignore_mismatched_sizes substitui a cabeça do ImageNet pela sua de num_classes
        self.backbone = ResNetForImageClassification.from_pretrained(
            self.__tag_name_huggingface__,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.freeze()

    def freeze(self):
        if self.freeze_backbone:
            # Congela a ResNet, mas mantém o 'classifier' treinável
            for param in self.backbone.resnet.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        # Extraímos o tensor de logits bruto do objeto da Hugging Face
        return self.backbone(pixel_values).logits