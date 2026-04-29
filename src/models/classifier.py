import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class MultimodalLinearProbe(nn.Module):
    """
    Implementa Linear Probing sobre backbones multimodais (CLIP/SigLIP).
    O backbone extrai features e a camada linear classifica.
    """
    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = model_name
        
        print(f"[*] Carregando backbone: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Congela os pesos do backbone para treinar apenas o classificador (Linear Probing)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Identifica dinamicamente a dimensão de saída (512 para base, 768 para large/siglip)
        hidden_size = self.backbone.config.vision_config.hidden_size \
                      if hasattr(self.backbone.config, 'vision_config') \
                      else self.backbone.config.hidden_size
                      
        print(f"[+] Hidden size detectado: {hidden_size}")
        
        # Cabeçalho de Classificação
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        # Passa a imagem pelo Vision Transformer do modelo
        outputs = self.backbone.vision_model(pixel_values=pixel_values)
        
        # Extrai o [CLS] token (representação global da imagem)
        pooled_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        
        # Classifica
        logits = self.classifier(pooled_output)
        return logits

class ModelFactory:
    """Factory Pattern para instanciar o modelo e seu processador associado."""
    
    @staticmethod
    def create(model_name: str, num_classes: int, freeze_backbone: bool = True):
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = MultimodalLinearProbe(model_name, num_classes, freeze_backbone)
        return model, processor