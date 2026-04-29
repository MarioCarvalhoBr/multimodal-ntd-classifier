source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 poetry run python src/run_experiment.py \
  --classes clinical_leprosy microscopy_chagas \
  --nets google/siglip-base-patch16-224 \
  --epochs 1 \
  --batch_size 32 \
  --num_workers 0 \
  --use_single_gpu

echo "Experiment finished"