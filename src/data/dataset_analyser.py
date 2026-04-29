import sys
from pathlib import Path

# Força a pasta 'src' a ser o diretório raiz para os imports
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import cv2
import numpy as np
import os
from typing import List, Dict
from tqdm import tqdm

from utils.logger import logger

class DatasetAnalyzer:
    """Analisa e limpa datasets identificando imagens vazias e arquivos corrompidos."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.splits = ['train', 'val', 'test']
        
        # Listas separadas para rastreabilidade fina
        self.black_images = []
        self.corrupted_images = []
        
        # Estatísticas
        self.stats = {
            "total_images": 0,
            "total_black": 0,
            "total_corrupted": 0,
            "split_counts": {s: 0 for s in self.splits},
            "class_counts": {}
        }

    def _check_image_status(self, img_path: Path) -> str:
        """
        Verifica o status da imagem.
        Retorna: 'ok', 'black' (toda preta) ou 'corrupted' (erro de leitura/0 bytes)
        """
        try:
            # 1. Verifica se o arquivo está vazio no disco (0 bytes)
            if img_path.stat().st_size == 0:
                return "corrupted"

            # 2. Tenta ler a imagem com o OpenCV
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "corrupted"
            
            # 3. Verifica se a imagem é completamente preta (ou ruído imperceptível)
            if np.max(img) <= 5: 
                return "black"
                
            return "ok"
        except Exception:
            # Qualquer erro de I/O ou decodificação cai aqui
            return "corrupted"

    def analyze(self):
        """Varre o dataset coletando estatísticas e classificando as imagens."""
        logger.info(f"[*] Iniciando análise de integridade em: {self.data_dir}")
        
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
                    
                    status = self._check_image_status(img_path)
                    
                    if status == "black":
                        self.stats["total_black"] += 1
                        self.black_images.append(img_path)
                    elif status == "corrupted":
                        self.stats["total_corrupted"] += 1
                        self.corrupted_images.append(img_path)

    def purge_invalid_images(self):
        """Remove fisicamente as imagens pretas e corrompidas do disco."""
        invalid_images = self.black_images + self.corrupted_images
        
        if not invalid_images:
            logger.info("[+] Nenhuma imagem inválida para remover. Dataset limpo!")
            return

        logger.info(f"[*] Purgando {len(invalid_images)} imagens inválidas do disco...")
        for img_path in tqdm(invalid_images, desc="Excluindo arquivos"):
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception as e:
                logger.error(f"Erro ao remover {img_path}: {e}")
        
        logger.info("[+] Limpeza concluída com sucesso.")

    def generate_report(self, output_file: str):
        """Gera o arquivo de sumário detalhado."""
        report_path = Path(output_dir) / output_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_invalid = self.stats["total_black"] + self.stats["total_corrupted"]
        pct_invalid = 0
        if self.stats["total_images"] > 0:
            pct_invalid = (total_invalid / self.stats["total_images"]) * 100

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("RELATÓRIO DE INTEGRIDADE DO DATASET\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. ESTATÍSTICAS GERAIS\n")
            f.write(f"Total Geral de Imagens: {self.stats['total_images']}\n")
            f.write(f"Total de Imagens Corrompidas (Erro/0 bytes): {self.stats['total_corrupted']}\n")
            f.write(f"Total de Imagens Vazias (Pretas): {self.stats['total_black']}\n")
            f.write(f"Porcentagem Total de Inválidas: {pct_invalid:.2f}%\n\n")
            
            f.write("2. DISTRIBUIÇÃO POR SPLIT\n")
            for split, count in self.stats["split_counts"].items():
                f.write(f"  - {split.upper()}: {count} imagens\n")
            f.write("\n")
            
            f.write("3. DISTRIBUIÇÃO POR CLASSE\n")
            for cls, count in self.stats["class_counts"].items():
                f.write(f"  - {cls}: {count} imagens\n")
            f.write("\n")
            
            f.write("="*50 + "\n")
            f.write("4. LISTA DE IMAGENS CORROMPIDAS (ERRO DE LEITURA)\n")
            f.write("="*50 + "\n")
            self._write_grouped_list(f, self.corrupted_images)
            
            f.write("\n" + "="*50 + "\n")
            f.write("5. LISTA DE IMAGENS VAZIAS (FUNDO PRETO)\n")
            f.write("="*50 + "\n")
            self._write_grouped_list(f, self.black_images)
                        
        logger.info(f"[+] Relatório gerado com sucesso em: {report_path}")

    def _write_grouped_list(self, file_handle, image_list):
        """Função auxiliar para escrever as listas de forma agrupada por pasta."""
        if not image_list:
            file_handle.write("Nenhuma imagem detectada nesta categoria.\n")
            return
            
        grouped_paths = {}
        for path in image_list:
            group_key = f"{path.parent.parent.name}/{path.parent.name}" 
            if group_key not in grouped_paths:
                grouped_paths[group_key] = []
            grouped_paths[group_key].append(path.name)
        
        for group, files in grouped_paths.items():
            file_handle.write(f"\n[{group}]\n")
            for file in files:
                file_handle.write(f"  └── {file}\n")

def main():
    parser = argparse.ArgumentParser(description="Analisa e limpa o dataset de imagens vazias/corrompidas.")
    parser.add_argument("--data_dir", type=str, default="dataset/processed/Dataset-NTD-V1", help="Caminho raiz do dataset processado")
    parser.add_argument("--purge-void", action="store_true", help="Ativa a remoção física das imagens corrompidas e pretas")
    args = parser.parse_args()

    global output_dir
    output_dir = Path("output/data_analysis")

    analyzer = DatasetAnalyzer(args.data_dir)
    
    analyzer.analyze()
    analyzer.generate_report("summary_dataset.txt")
    
    if args.purge_void:
        analyzer.purge_invalid_images()
    else:
        logger.info("\n[*] DICA: As imagens inválidas foram apenas listadas no relatório.")
        logger.info("[*] Para apagá-las do disco, execute o script com a flag: --purge-void")

if __name__ == "__main__":
    main()