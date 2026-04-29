import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path

class NTDVisionDataset(Dataset):
    """
    Dataset PyTorch para carregar as imagens de DTNs.
    Aplica processadores da Hugging Face e, opcionalmente, filtros customizados.
    """
    def __init__(self, root_dir: str, hf_processor, custom_preprocessor=None):
        self.root_dir = Path(root_dir)
        self.hf_processor = hf_processor
        self.custom_preprocessor = custom_preprocessor
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.rglob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        label = self.labels[idx]
        
        # Carregamento com OpenCV para compatibilidade com nosso preprocessor da Fase 1
        image_cv = cv2.imread(img_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Aplica remoção de pelos apenas se for imagem clínica (ex: clinical_leprosy)
        # Princípio Liskov: o preprocessor só é invocado se necessário
        if self.custom_preprocessor and "clinical" in img_path:
            image_cv = self.custom_preprocessor.process(image_cv)
            
        # Converte de volta para PIL Image, que é o formato nativo esperado pela Hugging Face
        image_pil = Image.fromarray(image_cv)
        
        # Processador Hugging Face (Redimensionamento, Normalização específica do CLIP/SigLIP)
        encoding = self.hf_processor(images=image_pil, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze(0) # Remove a dimensão do batch
        
        return pixel_values, torch.tensor(label, dtype=torch.long)