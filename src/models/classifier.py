import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import timm

# List todos os modelos disponíveis no timm para debug
print("Modelos disponíveis no timm:")
for model_name in timm.list_models():
    pass # print(f" - {model_name}")
    
# exit(0)

class MultimodalLinearProbe(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=True):
        super().__init__()
        
        if "efficientnet" in model_name or "vit_base_patch16_224" in model_name:
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
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Diagnóstico: confirma que o freeze funcionou
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            print(f"[DEBUG] Backbone '{model_name}': {trainable}/{total} parâmetros treináveis (esperado: 0)")
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        if self.is_timm:
            with torch.no_grad():
                features = self.backbone(pixel_values)  # input e backbone no mesmo device
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
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            
        model = MultimodalLinearProbe(model_name, num_classes, freeze_backbone)
        return model, processor