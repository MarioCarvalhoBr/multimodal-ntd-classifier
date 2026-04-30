source .venv/bin/activate
# Força o PyTorch a gerenciar a memória sem fragmentar (Vital para a EfficientNet e ResNet)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 poetry run python src/run_test.py \
  --classes microscopy_parasite_babesia microscopy_parasite_plasmodium microscopy_parasite_trichomonad microscopy_parasite_leishmania microscopy_parasite_rbcs microscopy_parasite_trypanosome microscopy_parasite_leukocyte microscopy_parasite_toxoplasma \
  --nets microsoft/resnet-50 \
  --batch_size 8 \
  --num_workers 0

echo "Teste concluído. Verifique os resultados e ajuste os parâmetros conforme necessário."