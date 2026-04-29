

import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.dataset import NTDDataset
from models.classifier import ModelFactory
from models.trainer import ModelTrainer as Trainer
from features.preprocessors import HairRemovalPreprocessor

def main():
    # 1. Configuração de Argumentos
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos Multimodais I CADTN")
    parser.add_argument(
        "--classes", 
        nargs="+", 
        help="Nomes das pastas das classes a treinar (ex: clinical_leprosy chagas_microscopy)",
        required=True
    )
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (sugerido 512 para 2x RTX 3080)")
    parser.add_argument("--num_workers", type=int, default=4, help="Número de workers para o DataLoader")
    args = parser.parse_args()

    # 2. Verificação de Segurança das Classes
    data_dir = Path("dataset/processed/Dataset-NTD-V1")
    train_dir = data_dir / "train"
    
    # Lista classes disponíveis no disco
    available_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    
    # Valida se as classes escolhidas existem
    for cls in args.classes:
        if cls not in available_classes:
            print(f"❌ ERRO: A classe '{cls}' não foi encontrada em {train_dir}")
            print(f"Classes disponíveis: {available_classes}")
            return

    print(f"✅ Validação concluída. Treinando para {len(args.classes)} classes: {args.classes}")

    # 3. Configuração de Hardware (Multi-GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"------>Usando device: {device}")
    print(f'Cuda disponível: {torch.cuda.is_available()}')
    print(f'Quantidade de GPUs: {torch.cuda.device_count()}')
    print(f'GPU atual: {torch.cuda.get_device_name(0)}')
    
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Num workers: {args.num_workers}")
    
    print(f"[*] Usando device: {device} ({torch.cuda.device_count()} GPUs detectadas)")
    print(f"[*] Classes selecionadas: {args.classes}")
    
    print(f"------>")

    MODELS_TO_TEST = [
        #"google/siglip-base-patch16-224",
        #"openai/clip-vit-base-patch16",
        "efficientnet_b3",
        #"vit_base_patch16_224"
    ]

    for model_name in MODELS_TO_TEST:
        print(f"\n" + "#"*60)
        print(f"[*] INICIANDO EXPERIMENTO: {model_name}")
        print("#"*60)

        # Criar modelo e processor
        model, processor = ModelFactory.create(model_name, num_classes=len(args.classes))
        
        # Ativar DataParallel para as 2x RTX 3080
        # if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)
        model.to(device)

        # Carregar datasets filtrados pelas classes escolhidas
        # Inicializa o preprocessor da Fase 1
        hair_preprocessor = HairRemovalPreprocessor()

        # Passa o preprocessor para o Dataset
        train_dataset = NTDDataset(
            data_dir / "train", 
            processor, 
            class_filter=args.classes, 
            custom_preprocessor=hair_preprocessor
        )
        val_dataset = NTDDataset(
            data_dir / "val", 
            processor, 
            class_filter=args.classes, 
            custom_preprocessor=hair_preprocessor
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        # Treinamento
        trainer = Trainer(model, device=device)
        save_path = f"output/best_model_{model_name.replace('/', '_')}.pt"
        
        trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=save_path)
        
        # Salvar curvas em PDF
        trainer.save_curves(model_name)

if __name__ == "__main__":
    main()