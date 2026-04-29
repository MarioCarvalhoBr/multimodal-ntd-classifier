source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 poetry run python src/run_test.py \
  --classes clinical_leprosy microscopy_chagas \
  --nets openai/clip-vit-base-patch16 \
  --batch_size 256 \
  --num_workers 0
