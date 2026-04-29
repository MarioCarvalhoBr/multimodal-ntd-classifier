source .venv/bin/activate
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 poetry run python src/run_experiment.py \
  --classes clinical_leprosy microscopy_chagas \
  --nets google/siglip-base-patch16-224 openai/clip-vit-base-patch16 \
  --epochs 1 \
  --batch_size 256 \
  --num_workers 0 \
  --use_single_gpu

  # microscopy_schistosomiasis