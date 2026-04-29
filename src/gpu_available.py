import torch


def main():
    # 1. Configurações gerais do experimento
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")
    print(f'Cuda disponível: {torch.cuda.is_available()}')
    print(f'Quantidade de GPUs: {torch.cuda.device_count()}')
    print(f'GPU atual: {torch.cuda.get_device_name(0)}')
    

if __name__ == "__main__":
    main()