import torch

from utils.logger import logger



def main():
    # 1. Configurações gerais do experimento
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando device: {device}")
    logger.info(f'Cuda disponível: {torch.cuda.is_available()}')
    logger.info(f'Quantidade de GPUs: {torch.cuda.device_count()}')
    logger.info(f'GPU atual: {torch.cuda.get_device_name(0)}')
    

if __name__ == "__main__":
    main()