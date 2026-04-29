from utils.logger import logger

# Importa as classes de forma independente
from .nets.cnn_resnet import CNN_ResNet50
from .nets.cnn_efficientnet import CNN_EfficientNetB3
from .nets.vit_base import ViT_BasePatch16_224
from .nets.vlm_clip import VLM_CLIP
from .nets.vlm_siglip import VLM_SigLIP

class ModelFactory:
    _registry = {}

    @classmethod
    def _register_all(cls):
        # Registra automaticamente todos os modelos pela sua tag do HuggingFace
        models = [CNN_ResNet50, CNN_EfficientNetB3, ViT_BasePatch16_224, VLM_CLIP, VLM_SigLIP]
        for model_cls in models:
            cls._registry[model_cls.__tag_name_huggingface__] = model_cls

    @classmethod
    def create(cls, model_name: str, num_classes: int, freeze_backbone: bool = True):
        if not cls._registry:
            cls._register_all()

        if model_name not in cls._registry:
            raise ValueError(f"❌ Modelo {model_name} não encontrado no registro. Disponíveis: {list(cls._registry.keys())}")

        # Identifica a classe a ser instanciada
        model_class = cls._registry[model_name]
        logger.info(f"[*] Construindo Classe de Rede: {model_class.__name__} | Tag HF: {model_name}")

        # Instancia a classe que já carrega o modelo e o processador internamente
        model_instance = model_class(num_classes=num_classes, freeze_backbone=freeze_backbone)
        
        return model_instance, model_instance.processor