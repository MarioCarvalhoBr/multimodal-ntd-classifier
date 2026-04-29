import argparse
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import NTDDataset
from models.classifier import ModelFactory
from models.trainer import ModelTrainer
from features.preprocessors import HairRemovalFilter # Nome correto da classe importada

def main():
    # 1. Configuração de Argumentos
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos Multimodais I CADTN")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--nets", nargs="+", required=True, help="Lista de modelos a serem testados (ex: efficientnet_b3 openai/clip-vit-base-patch16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_single_gpu", action="store_true", default=True, help="Força o uso de uma única GPU mesmo que múltiplas estejam disponíveis")
    args = parser.parse_args()
    
    # Print all arguments for clarity
    print("Configurações do Experimento:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # 2. Configuração de Hardware e Contexto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Inicializa o contexto CUDA forçadamente na thread principal
        #torch.cuda.init()
        print(f"✅ CUDA disponível. Usando GPU: {torch.cuda.get_device_name(0)}")
    
    # Plotar o numero da GPU utilizada
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        print(f"✅ GPU atual: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
        
    # 3. Verificação de Segurança das Classes
    data_dir = Path("dataset/processed/Dataset-NTD-V1")
    train_dir = data_dir / "train"
    
    available_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    for cls in args.classes:
        if cls not in available_classes:
            print(f"❌ ERRO: A classe '{cls}' não foi encontrada.")
            return

    print(f"✅ Treinando para {len(args.classes)} classes: {args.classes}")

    MODELS_TO_TEST = args.nets
    
    ALLOWED_MODELS = [
        "google/siglip-base-patch16-224",
        "openai/clip-vit-base-patch16",
        # Baselines Tradicionais de Visão
        "efficientnet_b3",             # Via timm
        "vit_base_patch16_224"         # Via timm
    ]
    if any(model not in ALLOWED_MODELS for model in MODELS_TO_TEST):
        print(f"❌ ERRO: Um ou mais modelos solicitados não estão na lista de modelos permitidos: {ALLOWED_MODELS}")
        return

    for model_name in MODELS_TO_TEST:
        print(f"\n" + "#"*60)
        print(f"[*] INICIANDO EXPERIMENTO: {model_name}")
        
        # Cria o modelo primeiro
        model, processor = ModelFactory.create(model_name, num_classes=len(args.classes))
        
        # Mova o modelo para a GPU ANTES de passar para o Trainer ou DataParallel
        model = model.to(device)

        if not args.use_single_gpu and torch.cuda.device_count() > 1:
            print("[*] Ativando DataParallel (Multi-GPU)")
            model = torch.nn.DataParallel(model)

        # Preprocessor da Fase 1
        hair_preprocessor = HairRemovalFilter()

        # Datasets
        train_dataset = NTDDataset(data_dir / "train", processor, args.classes, hair_preprocessor)
        val_dataset = NTDDataset(data_dir / "val", processor, args.classes, hair_preprocessor)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Treinamento (O ModelTrainer criará o otimizador internamente APÓS o modelo estar no device)
        trainer = ModelTrainer(model, device=str(device))
        save_path = f"models/saved/best_{model_name.replace('/', '_')}.pth"
        
        trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=save_path)
        trainer.save_curves(model_name)
if __name__ == "__main__":
    # Garante que o multiprocessing não corrompa o CUDA ao usar num_workers > 0
    try: 
        mp.set_start_method('spawn', force=True)
        print("✅ Multiprocessing configurado com 'spawn' para compatibilidade com CUDA.")
    except RuntimeError as e:
        print(f"❌ ERRO: {e}")
    main()
    
