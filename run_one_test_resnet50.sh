# Versão ultra-segura para depuração
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 poetry run python src/run_test.py \
  --classes microscopy_parasite_babesia microscopy_parasite_plasmodium microscopy_parasite_trichomonad microscopy_parasite_leishmania microscopy_parasite_rbcs microscopy_parasite_trypanosome microscopy_parasite_leukocyte microscopy_parasite_toxoplasma \
  --nets microsoft/resnet-50 \
  --batch_size 1 \
  --num_workers 0