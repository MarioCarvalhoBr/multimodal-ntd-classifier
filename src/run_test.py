import argparse
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import NTDDataset
from models.classifier import ModelFactory
from models.trainer import ModelTrainer as Trainer
from features.preprocessors import HairRemovalFilter

def main():
    # 1. Configuração de Argumentos
    parser = argparse.ArgumentParser(description="Avaliador Final de Modelos I CADTN")
    parser.add_argument("--classes", nargs="+", required=True, help="As mesmas classes usadas no treinamento")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamanho do batch para inferência")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_single_gpu", action="store_true")
    args = parser.parse_args()

    # 2. Hardware e Contexto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    # 3. Verificação de Dados
    data_dir = Path("dataset/processed/Dataset-NTD-V1")
    test_dir = data_dir / "test" # <<< ATENÇÃO AQUI: Lendo da pasta de teste
    
    if not test_dir.exists():
        print(f"❌ ERRO: A pasta de teste não foi encontrada em {test_dir}")
        return

    print(f"✅ Avaliando {len(args.classes)} classes: {args.classes}")

    # Certifique-se de que a lista de modelos bate com os que você treinou e salvou
    MODELS_TO_TEST = [
        # "efficientnet_b3",
        "openai/clip-vit-base-patch16"
    ]

    for model_name in MODELS_TO_TEST:
        print(f"\n" + "="*60)
        print(f"[*] INICIANDO AVALIAÇÃO DE TESTE: {model_name}")
        
        # Caminho do modelo salvo pelo run_experiment.py
        saved_model_path = f"models/saved/best_{model_name.replace('/', '_')}.pt"
        saved_model_path = f"models/saved/best_openai_clip-vit-base-patch16.pth"
                
        if not os.path.exists(saved_model_path):
            print(f"⚠️ AVISO: Pesos não encontrados em {saved_model_path}. Pule este modelo ou treine-o primeiro.")
            continue

        # 4. Criar o modelo
        model, processor = ModelFactory.create(model_name, num_classes=len(args.classes))
        
        # Carregar os pesos treinados ANTES de encapsular no DataParallel
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
        print(f"[+] Pesos carregados com sucesso: {saved_model_path}")
        
        # Mover para a GPU
        model = model.to(device)

        if not args.use_single_gpu and torch.cuda.device_count() > 1:
            print("[*] Ativando DataParallel para inferência rápida")
            model = torch.nn.DataParallel(model)

        # 5. Preprocessor da Fase 1
        hair_preprocessor = HairRemovalFilter()

        # 6. Carregar Dataset de Teste
        test_dataset = NTDDataset(test_dir, processor, args.classes, hair_preprocessor)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, # Não precisa embaralhar no teste
            num_workers=args.num_workers, 
            pin_memory=True
        )

        # 7. Executar Teste e Reportar
        trainer = Trainer(model, device=str(device))
        
        # Chama a função que já criamos no trainer.py para imprimir o Classification Report
        trainer.test_and_report(test_loader, target_names=args.classes)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()