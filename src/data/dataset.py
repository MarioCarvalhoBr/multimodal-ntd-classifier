import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import os

from utils.logger import logger


# Desativa o multithreading interno do OpenCV para evitar deadlocks com o DataLoader
# Isso é essencial para usar num_workers > 0 sem travamentos (segmentation faults)
cv2.setNumThreads(1)
os.environ["OMP_NUM_THREADS"] = "1"

class NTDDataset(Dataset):
    """
    Dataset customizado para Doenças Tropicais Negligenciadas (NTDs).
    Suporta filtragem de classes e pré-processamento específico (ex: remoção de pelos).
    """
    def __init__(self, root_dir, hf_processor, class_filter=None, custom_preprocessor=None):
        """
        Inicializa o dataset.
        
        Args:
            root_dir (str ou Path): Caminho raiz onde as pastas das classes estão (ex: 'data/train').
            hf_processor: O ImageProcessor da HuggingFace (ex: CLIPImageProcessor, AutoImageProcessor).
            class_filter (list, opcional): Lista de strings com os nomes das pastas das classes a incluir.
            custom_preprocessor (ImagePreprocessor, opcional): Instância de pré-processamento (ex: HairRemovalFilter).
        """
        self.root_dir = Path(root_dir)
        self.hf_processor = hf_processor
        self.custom_preprocessor = custom_preprocessor
        
        # Filtra as classes se solicitado, senão usa todas as pastas disponíveis
        if class_filter:
            self.classes = sorted(class_filter)
        else:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        
        # Varre as pastas selecionadas em busca de imagens
        for cls in self.classes:
            cls_path = self.root_dir / cls
            if cls_path.exists():
                # Ignora arquivos ocultos ou metadados, buscando apenas extensões de imagem comuns
                for img_path in cls_path.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.images.append((img_path, self.class_to_idx[cls]))
            else:
                logger.info(f"Aviso: O diretório para a classe '{cls}' não foi encontrado em {cls_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path_obj, label = self.images[idx]
        
        # Converte Path para string
        img_path_str = str(img_path_obj)
        
        # Carregamento 100% nativo com PIL para evitar problemas de multi-threading em C do OpenCV
        # Lembre-se, o PIL usa Lazy Loading, precisamos converter explicitamente para RGB.
        try:
            image_pil = Image.open(img_path_str).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Erro crítico: O PIL não conseguiu ler a imagem: {img_path_str}. Erro: {e}")
        
        # -------------------------------------------------------------
        # Pipeline Fase 1: Pré-processamento Baseado em Regras
        # Aplica a remoção de pelos apenas se o caminho indicar imagem clínica
        # -------------------------------------------------------------
        #if self.custom_preprocessor and "clinical" in img_path_str.lower():
        #    image_cv = self.custom_preprocessor.process(image_cv)
            
        # -------------------------------------------------------------
        # Pipeline Fase 2: Processamento Neural (HuggingFace)
        # Redimensionamento, CenterCrop, e Normalização (Z-Score)
        # -------------------------------------------------------------
        encoding = self.hf_processor(images=image_pil, return_tensors="pt")
        
        # O processor retorna um tensor no formato (Batch=1, Channels, Height, Width)
        # O squeeze(0) remove a dimensão de batch extra para o DataLoader
        pixel_values = encoding['pixel_values'].squeeze(0)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)