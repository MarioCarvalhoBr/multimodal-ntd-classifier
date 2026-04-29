import argparse
import torch
from torch.utils.data import DataLoader
from features.preprocessors import HairRemovalFilter # Nome correto da classe
from data.dataset import NTDDataset
from models.classifier import ModelFactory
from models.trainer import Trainer
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_single_gpu", action="store_true")
    args = parser.parse_args()

    data_dir = Path("dataset/processed/Dataset-NTD-V1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicializa o filtro da Fase 1
    hair_filter = HairRemovalFilter()

    # MODELS = ["efficientnet_b3", "vit_base_patch16_224", "openai/clip-vit-base-patch16"]
    MODELS = ["efficientnet_b3"]

    for model_name in MODELS:
        # 1. Criar Modelo e Processador
        model, processor = ModelFactory.create(model_name, num_classes=len(args.classes))
        
        # 2. Configuração Multi-GPU ou Single-GPU
        if not args.use_single_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # 3. MOVER PARA O DEVICE ANTES DE QUALQUER OUTRA OPERAÇÃO
        model = model.to(device)

        # 4. Datasets com Filtro e Preprocessor
        train_ds = NTDDataset(data_dir/"train", processor, args.classes, hair_filter)
        val_ds = NTDDataset(data_dir/"val", processor, args.classes, hair_filter)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

        # 5. Treinamento
        trainer = Trainer(model, device=device)
        
        save_path = f"output/best_model_{model_name.replace('/', '_')}.pt"
        trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=save_path)
        trainer.save_curves(model_name)

if __name__ == "__main__":
    main()