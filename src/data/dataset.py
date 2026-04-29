import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class NTDDataset(Dataset):
    def __init__(self, root_dir, hf_processor, class_filter=None, custom_preprocessor=None):
        self.root_dir = Path(root_dir)
        self.hf_processor = hf_processor
        self.custom_preprocessor = custom_preprocessor # Correção do erro aqui
        
        # Se houver filtro, usa apenas as pastas listadas. Se não, usa todas.
        if class_filter:
            self.classes = sorted(class_filter)
        else:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        
        for cls in self.classes:
            cls_path = self.root_dir / cls
            if cls_path.exists():
                for img_path in cls_path.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.png']:
                        self.images.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path_obj, label = self.images[idx]
        img_path_str = str(img_path_obj)
        
        # Carregamento robusto com OpenCV
        image_cv = cv2.imread(img_path_str)
        if image_cv is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem: {img_path_str}")
            
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Aplica remoção de pelos apenas se for imagem clínica
        if self.custom_preprocessor and "clinical" in img_path_str.lower():
            image_cv = self.custom_preprocessor.process(image_cv)
            
        image_pil = Image.fromarray(image_cv)
        
        # Processamento para o modelo (CLIP/SigLIP/EfficientNet)
        encoding = self.hf_processor(images=image_pil, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze(0)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)