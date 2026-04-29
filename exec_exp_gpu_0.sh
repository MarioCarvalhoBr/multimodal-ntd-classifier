source .venv/bin/activate
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 poetry run python src/run_experiment.py \
  --classes clinical_leprosy microscopy_chagas \
  --nets vit_base_patch16_224 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 0 \
  --use_single_gpu


# imprime o comando para debug
echo "Comando executado:"
echo "CUDA_VISIBLE_DEVICES=0 poetry run python src/run_experiment.py --classes clinical_leprosy microscopy_chagas --batch_size 16 --nets efficientnet_b3 --epochs 1 --num_workers 0 --use_single_gpu"