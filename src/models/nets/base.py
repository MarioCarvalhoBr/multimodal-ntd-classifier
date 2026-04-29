import torch.nn as nn

class BaseModel(nn.Module):
    """Classe base para todos os modelos de classificação."""
    __tag_name_huggingface__ = ""

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.processor = None
        self.backbone = None

    def freeze(self):
        """Método genérico para congelar os pesos. Deve ser sobreposto se a arquitetura exigir."""
        if self.freeze_backbone and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        raise NotImplementedError("O método forward deve ser implementado pelas subclasses.")