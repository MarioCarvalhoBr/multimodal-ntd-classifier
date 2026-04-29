import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import timm

class MultimodalLinearProbe(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=True):
        super().__init__()
        
        # Verifica se é um modelo timm ou HF
        if "efficientnet" in model_name or "vit_base_patch16_224" in model_name:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.is_timm = True
            
            # Descobre o tamanho do vetor de features via dummy pass
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                hidden_size = self.backbone(dummy).shape[1]
                
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.is_timm = False
            
            # Descobre o tamanho do vetor de features via dummy pass para HF (Bulletproof)
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                # Precisamos simular a mesma chamada que acontece no forward
                if hasattr(self.backbone, "get_image_features"):
                    out = self.backbone.get_image_features(dummy_input)
                else:
                    out = self.backbone(dummy_input).pooler_output
                hidden_size = out.shape[1]

        # Congela o backbone (Linear Probing)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Agora hidden_size será o valor exato, resolvendo o erro de matriz!
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        if self.is_timm:
            features = self.backbone(pixel_values)
        else:
            outputs = self.backbone.get_image_features(pixel_values) if hasattr(self.backbone, "get_image_features") else self.backbone(pixel_values).pooler_output
            features = outputs
            
        return self.classifier(features)

class ModelFactory:
    @staticmethod
    def create(model_name, num_classes, freeze_backbone=True):
        # Para modelos tradicionais de visão, usamos o processador do ViT como padrão de normalização
        if "efficientnet" in model_name or "vit_base_patch16_224" in model_name:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            
        model = MultimodalLinearProbe(model_name, num_classes, freeze_backbone)
        return model, processor