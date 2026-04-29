import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import sys

from utils.logger import logger


# Adiciona a pasta 'src' ao path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatasetAnalyzer:
    """Analisa e limpa datasets identificando imagens vazias/corrompidas."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.splits = ['train', 'val', 'test']
        self.black_images = []
        
        # Estatísticas
        self.stats = {
            "total_images": 0,
            "total_black": 0,
            "split_counts": {s: 0 for s in self.splits},
            "class_counts": {}
        }

    def _is_black_image(self, img_path: Path) -> bool:
        """Verifica se uma imagem é completamente preta (ou praticamente preta)."""
        try:
            # Lemos a imagem em escala de cinza
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True # Considera arquivo corrompido como 'preto/inválido'
            
            # Verifica se o valor máximo de pixel é 0 (ou muito próximo)
            # Usamos um limiar pequeno (ex: 5) para lidar com ruído de compressão JPEG
            if np.max(img) <= 5: 
                return True
                
            return False
        except Exception:
            return True

    def analyze(self):
        """Varre o dataset coletando estatísticas e identificando imagens pretas."""
        logger.info(f"[*] Iniciando análise do dataset em: {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.data_dir}")

        for split in self.splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
                
            classes = [d for d in split_dir.iterdir() if d.is_dir()]
            
            for cls_dir in classes:
                cls_name = cls_dir.name
                if cls_name not in self.stats["class_counts"]:
                    self.stats["class_counts"][cls_name] = 0
                    
                images = list(cls_dir.glob("*.*"))
                
                for img_path in tqdm(images, desc=f"Analisando {split}/{cls_name}", leave=False):
                    self.stats["total_images"] += 1
                    self.stats["split_counts"][split] += 1
                    self.stats["class_counts"][cls_name] += 1
                    
                    if self._is_black_image(img_path):
                        self.stats["total_black"] += 1
                        self.black_images.append(img_path)

    def purge_black_images(self):
        """Remove fisicamente as imagens pretas do disco."""
        if not self.black_images:
            logger.info("[+] Nenhuma imagem preta para remover.")
            return

        logger.info(f"[*] Removendo {len(self.black_images)} imagens vazias...")
        for img_path in tqdm(self.black_images, desc="Purgando imagens"):
            try:
                img_path.unlink()
            except Exception as e:
                logger.error(f"Erro ao remover {img_path}: {e}")
        
        logger.info("[+] Limpeza concluída.")

    def generate_report(self, output_file: str):
        """Gera o arquivo de sumário detalhado."""
        report_path = Path(output_dir) / output_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        pct_black = 0
        if self.stats["total_images"] > 0:
            pct_black = (self.stats["total_black"] / self.stats["total_images"]) * 100

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("RELATÓRIO DE INTEGRIDADE DO DATASET\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. ESTATÍSTICAS GERAIS\n")
            f.write(f"Total Geral de Imagens: {self.stats['total_images']}\n")
            f.write(f"Total de Imagens Pretas/Inválidas: {self.stats['total_black']}\n")
            f.write(f"Porcentagem de Imagens Inválidas: {pct_black:.2f}%\n\n")
            
            f.write("2. DISTRIBUIÇÃO POR SPLIT\n")
            for split, count in self.stats["split_counts"].items():
                f.write(f"  - {split.upper()}: {count} imagens\n")
            f.write("\n")
            
            f.write("3. DISTRIBUIÇÃO POR CLASSE\n")
            for cls, count in self.stats["class_counts"].items():
                f.write(f"  - {cls}: {count} imagens\n")
            f.write("\n")
            
            f.write("="*50 + "\n")
            f.write("4. LISTA DE IMAGENS PRETAS/INVÁLIDAS\n")
            f.write("="*50 + "\n")
            
            if not self.black_images:
                f.write("Nenhuma imagem inválida encontrada.\n")
            else:
                # Agrupa os caminhos para exibição organizada
                grouped_paths = {}
                for path in self.black_images:
                    # Ex: train/clinical_leprosy
                    group_key = f"{path.parent.parent.name}/{path.parent.name}" 
                    if group_key not in grouped_paths:
                        grouped_paths[group_key] = []
                    grouped_paths[group_key].append(path.name)
                
                for group, files in grouped_paths.items():
                    f.write(f"\n[{group}]\n")
                    for file in files:
                        f.write(f"  └── {file}\n")
                        
        logger.info(f"[+] Relatório gerado com sucesso em: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analisa e limpa o dataset de imagens vazias/corrompidas.")
    parser.add_argument("--data_dir", type=str, default="dataset/processed/Dataset-NTD-V1", help="Caminho raiz do dataset processado")
    parser.add_argument("--purge-void", action="store_true", help="Ativa a remoção física das imagens detectadas como pretas/corrompidas")
    args = parser.parse_args()

    # O output do relatório será salvo na pasta de análise de dados
    global output_dir
    output_dir = Path("output/data_analysis")

    analyzer = DatasetAnalyzer(args.data_dir)
    
    # 1. Varre o dataset
    analyzer.analyze()
    
    # 2. Gera o relatório (antes de apagar, para ter o registro histórico do que estava ruim)
    analyzer.generate_report("summary_dataset.txt")
    
    # 3. Apaga se a flag for acionada
    if args.purge_void:
        analyzer.purge_black_images()
    else:
        logger.info("\n[*] DICA: As imagens pretas foram apenas listadas no relatório.")
        logger.info("[*] Para apagá-las do disco, execute o script com a flag: --purge-void")

if __name__ == "__main__":
    main()
    
# Use example without argparse (for testing purposes): 
# poetry run python src/data/dataset_analyser.py

# Use example with argparse: 
# poetry run python src/data/dataset_analyser.py --purge-void
