import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
import shutil
from torch.utils.data import DataLoader

from data.dataset import NTDDataset
from models.factory import ModelFactory
from models.trainer import ModelTrainer
from features.preprocessors import HairRemovalFilter
from utils.logger import logger, log_file
from config.config import load_config

settings = load_config()

def main():
    # 1. Configuração de Argumentos
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos Multimodais I CADTN")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--nets", nargs="+", required=True, help="Lista de modelos a serem testados (ex: google/efficientnet-b3 openai/clip-vit-base-patch16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_single_gpu", action="store_true", default=True, help="Força o uso de uma única GPU mesmo que múltiplas estejam disponíveis")
    args = parser.parse_args()
    
    logger.info("Configurações do Experimento:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # 2. Configuração de Hardware e Contexto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.enabled = False
        logger.info(f"✅ CUDA disponível. Usando GPU: {torch.cuda.get_device_name(0)}")
    
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        logger.info(f"✅ GPU atual: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
        
    # 3. Verificação de Segurança das Classes
    data_dir = Path("dataset/processed/Dataset-NTD-V1")
    train_dir = data_dir / "train"
    
    available_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    for cls in args.classes:
        if cls not in available_classes:
            logger.info(f"❌ ERRO: A classe '{cls}' não foi encontrada.")
            return

    logger.info(f"✅ Treinando para {len(args.classes)} classes: {args.classes}")

    MODELS_TO_TEST = args.nets
    
    ALLOWED_MODELS = settings.ALLOWED_MODELS
    if any(model not in ALLOWED_MODELS for model in MODELS_TO_TEST):
        logger.info(f"❌ ERRO: Um ou mais modelos solicitados não estão na lista de modelos permitidos: {ALLOWED_MODELS}")
        return

    for model_name in MODELS_TO_TEST:
        logger.info(f"\n" + "#"*60)
        logger.info(f"[*] INICIANDO EXPERIMENTO: {model_name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  
            logger.info(f"✅ Memória GPU limpa antes de iniciar o modelo '{model_name}'.")
            mem_free = torch.cuda.mem_get_info()[0] / 1024**3
            mem_total = torch.cuda.mem_get_info()[1] / 1024**3
            logger.info(f"GPU Memory: {mem_free:.1f}GB livre de {mem_total:.1f}GB total")
        
        # Criação do modelo e processador utilizando a nova Factory
        model, processor = ModelFactory.create(model_name, num_classes=len(args.classes))
        
        model = model.to(device)

        if not args.use_single_gpu and torch.cuda.device_count() > 1:
            logger.info("[*] Ativando DataParallel (Multi-GPU)")
            model = torch.nn.DataParallel(model)

        # Preprocessor da Fase 1
        hair_preprocessor = HairRemovalFilter()

        # Datasets
        train_dataset = NTDDataset(data_dir / "train", processor, args.classes, hair_preprocessor)
        val_dataset = NTDDataset(data_dir / "val", processor, args.classes, hair_preprocessor)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Treinamento
        trainer = ModelTrainer(model, device=str(device))
        
        # Formatação do nome para o diretório e arquivo salvos
        model_tag = model_name.replace("/", "_")
        output_dir = Path(f"output/results/{model_tag}/saved_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = f"{output_dir}/best_{model_tag}.pth"
        
        trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=save_path)
        trainer.save_curves(model_name)

        output_dir_log = Path(f"output/results/{model_tag}")
        output_dir_log.mkdir(parents=True, exist_ok=True)
        if log_file:
            shutil.copy2(log_file, output_dir_log / "log-train.log")
            logger.info(f"[+] Log copiado para: {output_dir_log / 'log-train.log'}")

if __name__ == "__main__":
    try: 
        mp.set_start_method('spawn', force=True)
        logger.info("✅ Multiprocessing configurado com 'spawn' para compatibilidade com CUDA.")
    except RuntimeError as e:
        logger.info(f"❌ ERRO: {e}")
    main()