import os
import shutil
import kagglehub
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from typing import List, Dict

# Carrega as variáveis do .env
load_dotenv()

class KaggleDownloader:
    """Responsável exclusivo por baixar datasets do Kaggle."""
    
    @staticmethod
    def download(dataset_handle: str) -> Path:
        print(f"[*] Baixando dataset: {dataset_handle}...")
        path = kagglehub.dataset_download(dataset_handle)
        print(f"[+] Download concluído. Salvo em: {path}")
        return Path(path)

class NTDDatasetBuilder:
    """Responsável por unificar os dados baixados e criar os splits (Train/Val/Test)."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.splits = ['train', 'val', 'test']
        
    def _create_directories(self, classes: List[str]):
        """Cria a estrutura de pastas: output_dir/split/class/"""
        for split in self.splits:
            for cls in classes:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def build_splits(self, source_path: Path, target_class: str, extensions: tuple = ('.jpg', '.png', '.jpeg')):
        """
        Varre o source_path, coleta imagens, faz o split 70/15/15 e copia para o Dataset-NTD-V1.
        Nota: Devido à heterogeneidade dos datasets do Kaggle, essa função assume 
        que vamos extrair todas as imagens de um diretório fonte e rotulá-las.
        """
        print(f"[*] Processando classe '{target_class}' a partir de {source_path}")
        
        # Coleta todas as imagens de forma recursiva
        images = []
        for ext in extensions:
            images.extend(list(source_path.rglob(f"*{ext}")))
            images.extend(list(source_path.rglob(f"*{ext.upper()}")))
            
        if not images:
            print(f"[!] Aviso: Nenhuma imagem encontrada para a classe {target_class}.")
            return

        # Stratified Split 70/15/15 (usando placeholder para os labels)
        labels = [target_class] * len(images)
        
        # Train (70%) e Temp (30%)
        X_train, X_temp, _, y_temp = train_test_split(
            images, labels, test_size=0.30, random_state=42
        )
        # Val (15%) e Test (15%)
        X_val, X_test, _, _ = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )

        splits_map = {'train': X_train, 'val': X_val, 'test': X_test}

        self._create_directories([target_class])

        # Copia os arquivos
        for split_name, img_paths in splits_map.items():
            for img_path in img_paths:
                dest = self.output_dir / split_name / target_class / img_path.name
                # Evita sobrescrever arquivos com mesmo nome gerando sufixo
                if dest.exists():
                    dest = dest.with_name(f"{img_path.stem}_{os.urandom(4).hex()}{img_path.suffix}")
                shutil.copy2(img_path, dest)
                
        print(f"[+] {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test imagens alocadas.")

def main():
    # 1. Configurações
    target_dir = os.getenv("PROCESSED_DATA_DIR", "dataset/processed/Dataset-NTD-V1")
    builder = NTDDatasetBuilder(output_dir=target_dir)
    
    # 2. Mapeamento dos Datasets (Handle do Kaggle -> Nome da Classe alvo)
    # Como os datasets possuem subpastas internas, você pode ajustar este mapeamento 
    # conforme explora o que foi baixado no cache.
    datasets_to_fetch = {
        "orvile/leprosy-chronic-wound-images-co2wounds-v2": "clinical_leprosy",
        "ahmedxc4/parasite-dataset": "microscopy_parasites_general",
        "andrpereira157/trypanosoma-cruzi-microscopy-detection-dataset": "microscopy_chagas",
        "mohaliy2016/ai4ntd-p1-5v2": "microscopy_schistosomiasis"
    }

    # 3. Execução da Pipeline
    for handle, target_class in datasets_to_fetch.items():
        # Download (ou uso do cache)
        raw_path = KaggleDownloader.download(handle)
        
        # Processamento e Split
        builder.build_splits(source_path=raw_path, target_class=target_class)

    print(f"\n[🚀] Dataset-NTD-V1 construído com sucesso em: {target_dir}")

if __name__ == "__main__":
    main()