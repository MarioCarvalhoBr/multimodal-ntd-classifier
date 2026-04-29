#!/bin/bash
# Ativa o ambiente virtual
source .venv/bin/activate

# Força o PyTorch a gerenciar a memória sem fragmentar (Vital para a EfficientNet e ResNet)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Lista de modelos (Use as tags exatas do Hugging Face)
MODELS=(
    "microsoft/resnet-50"
    "google/efficientnet-b3"
    "google/vit-base-patch16-224"
)

# Configurações globais
CLASSES="clinical_leprosy microscopy_chagas microscopy_schistosomiasis"
EPOCHS=4
NUM_WORKERS=0 # Mantenha 0 para estabilidade
CUDA_DEVICE=0 # GPU a ser utilizada (0, 1, etc.)

echo "=========================================================="
echo " INICIANDO BATERIA DE EXPERIMENTOS (TREINO + TESTE)"
echo "=========================================================="

for MODEL in "${MODELS[@]}"
do
    echo "\n\n>>> Processando Modelo: $MODEL <<<"
    
    # Define um batch size seguro dependendo se é CNN ou Transformer
    if [[ "$MODEL" == *"efficientnet"* || "$MODEL" == *"resnet"* ]]; then
        BATCH_SIZE=16  # Mais conservador para CNNs pesadas na memória
    else
        BATCH_SIZE=32 # Transformers lidam melhor com batches maiores
    fi

    # 1. Limpa a GPU rigorosamente antes de cada modelo
    echo "Limpando VRAM..."
    # sudo fuser -k -9 /dev/nvidia0 || true
    sleep 2

    # 2. Executa o TREINAMENTO
    echo "[TREINO] $MODEL (Epochs: $EPOCHS, Batch: $BATCH_SIZE)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE poetry run python src/run_experiment.py \
      --classes $CLASSES \
      --nets $MODEL \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --num_workers $NUM_WORKERS \
      --use_single_gpu
      
    # 3. Executa o TESTE logo em seguida
    echo "[TESTE] $MODEL"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE poetry run python src/run_test.py \
      --classes $CLASSES \
      --nets $MODEL \
      --batch_size $BATCH_SIZE \
      --num_workers $NUM_WORKERS \
      --use_single_gpu
      
done

echo "=========================================================="
echo "🎉 TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO! 🎉"
echo "=========================================================="