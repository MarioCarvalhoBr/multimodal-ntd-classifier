import argparse
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader

import shutil

from data.dataset import NTDDataset
from models.factory import ModelFactory
from models.trainer import ModelTrainer as Trainer
from features.preprocessors import HairRemovalFilter
from utils.logger import logger, log_file
from config.config import load_config


settings = load_config()

def main():
    # 1. Configuração de Argumentos
    parser = argparse.ArgumentParser(description="Avaliador Final de Modelos I CADTN")
    parser.add_argument("--classes", nargs="+", required=True, help="As mesmas classes usadas no treinamento")
    parser.add_argument("--nets", nargs="+", required=True, help="Lista de modelos a serem testados (ex: google/efficientnet-b3 openai/clip-vit-base-patch16)")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamanho do batch para inferência")
    parser.add_argument("--num_workers", type=int, default=0) # Garantindo 0 para evitar deadlocks
    parser.add_argument("--use_single_gpu", action="store_true")
    args = parser.parse_args()

    # 2. Hardware e Contexto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA disponível. Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️ CUDA não disponível. Usando CPU, o que pode ser muito lento para modelos grandes!")
        
    logger.info(f"✅ Dispositivo selecionado: {device}")

    # 3. Verificação de Dados
    data_dir = Path("dataset/processed/Dataset-NTD-V1")
    test_dir = data_dir / "test" 
    
    if not test_dir.exists():
        logger.info(f"❌ ERRO: A pasta de teste não foi encontrada em {test_dir}")
        return

    logger.info(f"✅ Avaliando {len(args.classes)} classes: {args.classes}")
    logger.info(f"✅ Modelos a serem testados: {args.nets}")

    MODELS_TO_TEST = args.nets
    ALLOWED_MODELS = settings.ALLOWED_MODELS
    
    if any(model not in ALLOWED_MODELS for model in MODELS_TO_TEST):
        logger.info(f"❌ ERRO: Um ou mais modelos solicitados não estão na lista de modelos permitidos: {ALLOWED_MODELS}")
        return

    for model_name in MODELS_TO_TEST:
        logger.info(f"\n" + "="*60)
        logger.info(f"[*] INICIANDO AVALIAÇÃO DE TESTE: {model_name}")
        
        # Caminho do modelo salvo pelo run_experiment.py
        model_tag = model_name.replace("/", "_")
        output_dir = Path(f"output/results/{model_tag}/saved_model")
        
        saved_model_path = f"{output_dir}/best_{model_tag}.pth"
                
        if not os.path.exists(saved_model_path):
            logger.info(f"⚠️ AVISO: Pesos não encontrados em {saved_model_path}. Pule este modelo ou treine-o primeiro.")
            continue

        # 4. Criar o modelo utilizando a nova Factory
        model, processor = ModelFactory.create(model_name, num_classes=len(args.classes))
        
        # Carregar os pesos treinados
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
        logger.info(f"[+] Pesos carregados com sucesso: {saved_model_path}")
        
        # Mover para a GPU
        model = model.to(device)

        if not args.use_single_gpu and torch.cuda.device_count() > 1:
            logger.info("[*] Ativando DataParallel para inferência rápida")
            model = torch.nn.DataParallel(model)

        # 5. Preprocessor da Fase 1
        hair_preprocessor = HairRemovalFilter()

        # 6. Carregar Dataset de Teste
        test_dataset = NTDDataset(test_dir, processor, args.classes, hair_preprocessor)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )

        # 7. Executar Teste e Reportar
        trainer = Trainer(model, device=str(device))
        trainer.test_and_report(test_loader, target_names=args.classes, model_name=model_name)
        logger.info(f"[*] Avaliação de teste concluída para: {model_name}")
        
        output_dir_log = Path(f"output/results/{model_tag}")
        output_dir_log.mkdir(parents=True, exist_ok=True)
        if log_file:
            shutil.copy2(log_file, output_dir_log / "log-test.log")
            logger.info(f"[+] Log copiado para: {output_dir_log / 'log-test.log'}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()