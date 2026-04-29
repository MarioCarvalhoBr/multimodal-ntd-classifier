source .venv/bin/activate
# Força o PyTorch a gerenciar a memória sem fragmentar (Vital para a EfficientNet e ResNet)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 poetry run python src/run_test.py \
  --classes clinical_leprosy microscopy_chagas microscopy_schistosomiasis microscopy_parasites_general \
  --nets google/siglip-base-patch16-224 \
  --batch_size 64 \
  --num_workers 0

echo "Teste concluído para SigLIP. Verifique os resultados e ajuste os parâmetros conforme necessário."