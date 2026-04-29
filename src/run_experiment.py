import os
import sys

# Adiciona a pasta 'src' ao path do Python para ele encontrar os módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from data.dataset import NTDVisionDataset
from features.preprocessors import HairRemovalFilter
from models.classifier import ModelFactory
from models.trainer import ModelTrainer

def main():
    # 1. Configurações gerais do experimento
    data_dir = os.getenv("PROCESSED_DATA_DIR", "dataset/processed/Dataset-NTD-V1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    epochs = 10
    
    # Modelos que você listou para comparar
    models_to_test = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "google/siglip-base-patch16-224",
        # "openai/clip-vit-large-patch14" # (Descomente se tiver VRAM suficiente)
    ]

    # Instanciando nosso filtro do Phase 1
    # Observação: será ativado dentro do Dataset apenas para as imagens clínicas
    preprocessor = HairRemovalFilter(kernel_size=(17, 17), threshold=10)

    # 2. Loop de Experimentos
    for model_name in models_to_test:
        print("\n" + "#"*60)
        print(f"[*] INICIANDO EXPERIMENTO: {model_name}")
        print("#"*60)

        # Usando nossa Factory para criar o Modelo e Processador
        model, processor = ModelFactory.create(
            model_name=model_name, 
            num_classes=4, # Leprosy, Chagas, Parasites_Gen, Schistosomiasis
            freeze_backbone=True # Linear Probing
        )

        # Criando os Datasets com o processador específico deste modelo
        train_dataset = NTDVisionDataset(f"{data_dir}/train", processor, preprocessor)
        val_dataset = NTDVisionDataset(f"{data_dir}/val", processor, preprocessor)
        test_dataset = NTDVisionDataset(f"{data_dir}/test", processor, preprocessor)

        # Classes extraídas automaticamente pelas pastas
        class_names = train_dataset.classes

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Configurando o Treinador
        trainer = ModelTrainer(model=model, device=device, learning_rate=1e-3)

        # Caminho para salvar o melhor peso
        safe_model_name = model_name.replace("/", "_")
        save_path = f"models/saved/{safe_model_name}_best.pth"

        # Rodando o Fit (Treinamento e Validação)
        trainer.fit(train_loader, val_loader, epochs=epochs, save_path=save_path)

        # Carregando o melhor peso para o Teste Final
        model.load_state_dict(torch.load(save_path))
        
        # Relatório Final
        trainer.test_and_report(test_loader, target_names=class_names)

if __name__ == "__main__":
    main()