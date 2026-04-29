source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 poetry run python src/run_experiment.py --classes clinical_leprosy microscopy_chagas --nets efficientnet_b3 --epochs 1 --num_workers 4 --use_single_gpu