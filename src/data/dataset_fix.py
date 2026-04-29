import sys
from pathlib import Path

# Força a pasta 'src' a ser o diretório raiz para os imports (resolve o erro do utils.logger)
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
from PIL import Image
from tqdm import tqdm
from utils.logger import logger

def split_and_replace_collages(data_dir: str, target_class: str):
    base_path = Path(data_dir)
    splits = ['train', 'val', 'test']
    
    total_processed = 0
    total_created = 0

    for split in splits:
        class_dir = base_path / split / target_class
        if not class_dir.exists():
            continue

        # Busca imagens, mas ignora as que já têm "_part_" no nome para evitar recortes infinitos
        images = [p for p in class_dir.glob("*.*") 
                  if p.suffix.lower() in ['.jpg', '.jpeg', '.png'] and "_part_" not in p.name]
        
        if not images:
            logger.info(f"[*] Nenhuma imagem mosaico original encontrada em {split}/{target_class}.")
            continue

        logger.info(f"[*] Processando {len(images)} mosaicos em {split}/{target_class}...")

        for img_path in tqdm(images, desc=f"Fatiando {split}"):
            try:
                # Usamos block 'with' para garantir que o arquivo seja liberado da RAM
                with Image.open(img_path) as img:
                    img.load() # Força o carregamento para desvincular do arquivo físico
                    width, height = img.size
                    
                    # Calcula as dimensões exatas de cada "quadrado" (3 colunas, 2 linhas)
                    w_step = width // 3
                    h_step = height // 2
                    
                    part_num = 1
                    for row in range(2):
                        for col in range(3):
                            # Coordenadas: (esquerda, topo, direita, base)
                            left = col * w_step
                            upper = row * h_step
                            # Ajuste fino para não perder 1 pixel de borda caso a divisão não seja exata
                            right = width if col == 2 else (col + 1) * w_step
                            lower = height if row == 1 else (row + 1) * h_step
                            
                            # Recorta o quadrante
                            crop_img = img.crop((left, upper, right, lower))
                            
                            # Salva a nova imagem
                            new_name = f"{img_path.stem}_part_{part_num}{img_path.suffix}"
                            new_path = class_dir / new_name
                            crop_img.save(new_path)
                            
                            total_created += 1
                            part_num += 1

                # Apaga o mosaico original com segurança
                img_path.unlink()
                total_processed += 1

            except Exception as e:
                logger.error(f"Erro ao fatiar a imagem {img_path}: {e}")

    logger.info("\n" + "="*50)
    logger.info("[🚀] OPERAÇÃO DE RECORTES CONCLUÍDA!")
    logger.info(f"  - Mosaicos originais destruídos: {total_processed}")
    logger.info(f"  - Novas imagens limpas geradas: {total_created}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide imagens mosaico (2x3) da Esquistossomose.")
    parser.add_argument("--data_dir", type=str, default="dataset/processed/Dataset-NTD-V1")
    parser.add_argument("--target_class", type=str, default="microscopy_schistosomiasis")
    args = parser.parse_args()

    split_and_replace_collages(args.data_dir, args.target_class)