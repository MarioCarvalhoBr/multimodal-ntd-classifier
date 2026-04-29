import sys
from pathlib import Path

# Força a pasta 'src' a ser o diretório raiz para os imports
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
import os
import shutil
import kagglehub
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from typing import List, Dict

from utils.logger import logger

# Carrega as variáveis do .env[cite: 8]
load_dotenv()

class KaggleDownloader:
    """Responsável exclusivo por baixar datasets do Kaggle[cite: 8]."""
    
    @staticmethod
    def download(dataset_handle: str) -> Path:
        logger.info(f"[*] Baixando dataset: {dataset_handle}...")
        path = kagglehub.dataset_download(dataset_handle)
        logger.info(f"[+] Download concluído. Salvo em: {path}")
        return Path(path)

class NTDDatasetBuilder:
    """Responsável por unificar os dados baixados e criar os splits (Train/Val/Test)[cite: 8]."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.splits = ['train', 'val', 'test']
        
    def _create_directories(self, classes: List[str]):
        """Cria a estrutura de pastas: output_dir/split/class/[cite: 8]"""
        for split in self.splits:
            for cls in classes:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def build_splits(self, source_path: Path, target_class: str, extensions: tuple = ('.jpg', '.png', '.jpeg')):
        """
        Varre o source_path, coleta imagens, faz o split 70/15/15 e copia para o Dataset-NTD-V1[cite: 8].
        """
        logger.info(f"[*] Processando classe '{target_class}' a partir de {source_path}")
        
        # Coleta todas as imagens de forma recursiva[cite: 8]
        images = []
        for ext in extensions:
            images.extend(list(source_path.rglob(f"*{ext}")))
            images.extend(list(source_path.rglob(f"*{ext.upper()}")))
            
        if not images:
            logger.info(f"[!] Aviso: Nenhuma imagem encontrada para a classe {target_class}.")
            return

        # Stratified Split 70/15/15[cite: 8]
        labels = [target_class] * len(images)
        
        # Train (70%) e Temp (30%)[cite: 8]
        X_train, X_temp, _, y_temp = train_test_split(
            images, labels, test_size=0.30, random_state=42
        )
        # Val (15%) e Test (15%)[cite: 8]
        X_val, X_test, _, _ = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )

        splits_map = {'train': X_train, 'val': X_val, 'test': X_test}

        self._create_directories([target_class])

        # Copia os arquivos[cite: 8]
        for split_name, img_paths in splits_map.items():
            for img_path in img_paths:
                dest = self.output_dir / split_name / target_class / img_path.name
                if dest.exists():
                    dest = dest.with_name(f"{img_path.stem}_{os.urandom(4).hex()}{img_path.suffix}")
                shutil.copy2(img_path, dest)
                
        logger.info(f"[+] {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test imagens alocadas.")

def main():
    # 1. Configurações
    target_dir = os.getenv("PROCESSED_DATA_DIR", "dataset/processed/Dataset-NTD-V1")
    
    if Path(target_dir).exists():
        logger.info(f"[*] Limpando diretório antigo em {target_dir}...")
        shutil.rmtree(target_dir)
        
    builder = NTDDatasetBuilder(output_dir=target_dir)
    
    # 2. Configuração do Parasite Dataset
    parasite_handle = "ahmedxc4/parasite-dataset"
    ntd_parasite_folders = [
        "Babesia", "Leishmania", "Leukocyte", "Plasmodium", 
        "RBCs", "Toxoplasma", "Trichomonad", "Trypanosome"
    ]

    # 3. Execução para o Parasite Dataset
    raw_path = KaggleDownloader.download(parasite_handle)
    
    # --- CORREÇÃO AQUI: Localiza a subpasta real (geralmente 'H0') ---
    # Buscamos a primeira pasta dentro do download que contenha as nossas classes
    actual_source_path = raw_path
    for root, dirs, files in os.walk(raw_path):
        if "Babesia" in dirs:
            actual_source_path = Path(root)
            logger.info(f"[+] Diretório real das imagens encontrado em: {actual_source_path}")
            break
    # ---------------------------------------------------------------
    
    for folder in ntd_parasite_folders:
        folder_path = actual_source_path / folder
        if folder_path.exists():
            target_class_name = f"microscopy_parasite_{folder.lower()}"
            builder.build_splits(source_path=folder_path, target_class=target_class_name)
        else:
            logger.info(f"[!] Pasta {folder} não encontrada em {folder_path}")

    logger.info(f"\n[🚀] Dataset-NTD-V1 reconstruído com sucesso em: {target_dir}")
if __name__ == "__main__":
    main()
    
# Use examples:
# python src/data/make_dataset.py