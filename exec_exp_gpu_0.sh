source .venv/bin/activate
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 poetry run python src/run_experiment.py \
  --classes clinical_leprosy microscopy_chagas microscopy_schistosomiasis \
  --nets resnet50 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 0 \
  --use_single_gpu