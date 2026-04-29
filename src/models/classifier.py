
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import timm

from utils.logger import logger


# List todos os modelos disponíveis no timm para debug
logger.info("Modelos disponíveis no timm:")
for model_name in timm.list_models():
    pass # logger.info(f" - {model_name}")
    
# exit(0)

class MultimodalLinearProbe(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        if "resnet50" in model_name:
            import torchvision.models as models
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the classification head by setting it to Identity
            self.backbone.fc = nn.Identity()
            self.is_timm = "torchvision"
            
            # Dummy pass na CPU antes de qualquer movimentação
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                hidden_size = self.backbone(dummy).shape[1]
                
        elif "efficientnet" in model_name or "vit_base_patch16_224" in model_name:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.is_timm = True
            
            # Dummy pass na CPU antes de qualquer movimentação
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                hidden_size = self.backbone(dummy).shape[1]
                
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.backbone = self.backbone.cpu()  # ← Mesma garantia para HF
            self.is_timm = False
            
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                if hasattr(self.backbone, "get_image_features"):
                    out = self.backbone.get_image_features(dummy_input)
                else:
                    out = self.backbone(dummy_input).pooler_output
                hidden_size = out.shape[1]

        if freeze_backbone:
            # Coloca forçosamente em eval()
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # If timm or torchvision, we should make sure BatchNorm layers don't track running stats
            if self.is_timm == True or self.is_timm == "torchvision":
                for m in self.backbone.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
                        m.track_running_stats = False

            # Diagnóstico: confirma que o freeze funcionou
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(f"[DEBUG] Backbone '{model_name}': {trainable}/{total} parâmetros treináveis (esperado: 0)")
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        if self.is_timm == True or self.is_timm == "torchvision":
            # Em modo frozen, a arquitetura deve estar em eval() para BatchNorm não computar specs
            if self.freeze_backbone:
                self.backbone.eval()
                
            # Desliga autograd ao longo do modelo frozen para prevenir hooks que levam a memory leaks
            with torch.set_grad_enabled(not self.freeze_backbone):
                # timm model num_classes=0 means it outputs a pooled tensor, not a spatial map
                # So we simply pass it through the regular forward method
                features = self.backbone(pixel_values)
        else:
            with torch.no_grad():
                outputs = (
                    self.backbone.get_image_features(pixel_values)
                    if hasattr(self.backbone, "get_image_features")
                    else self.backbone(pixel_values).pooler_output
                )
                features = outputs
        
        # Garante que o classifier está no mesmo device que as features
        self.classifier = self.classifier.to(features.device)
        return self.classifier(features)
class ModelFactory:
    @staticmethod
    def create(model_name, num_classes, freeze_backbone=True):
        # Para modelos tradicionais de visão, usamos o processador do ViT como padrão de normalização
        if "efficientnet" in model_name or "vit_base_patch16_224" in model_name:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        if "resnet50" in model_name:
            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            
        model = MultimodalLinearProbe(model_name, num_classes, freeze_backbone)
        return model, processor