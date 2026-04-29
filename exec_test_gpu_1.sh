source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 poetry run python src/run_test.py \
  --classes clinical_leprosy microscopy_chagas \
  --nets efficientnet_b3 \
  --num_workers 0