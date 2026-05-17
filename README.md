# Multimodal Vision-Language Classifier for Neglected Tropical Diseases

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-ffcc00)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

A research proof-of-concept benchmarking frozen Vision-Language Model (VLM) and CNN backbones via **linear probing** for microscopy image classification of parasites that cause Neglected Tropical Diseases (NTDs). Includes a **multimodal late-fusion** pipeline combining a fine-tuned linear probe with a Qdrant vector database for image and text retrieval. Developed and submitted to the **I Workshop de Computação Aplicada às Doenças Tropicais Negligenciadas (CADTN)**. See more details in the [website](https://cadtn.dotlabbrazil.com.br/).

---

## Screenshots

| | |
|---|---|
| ![Figure 1](assets/FIGURE_01.png) | ![Figure 2](assets/FIGURE_02.png) |
| ![Figure 3](assets/FIGURE_03.png) | ![Figure 4](assets/FIGURE_04.png) |

---

## Overview

The pipeline classifies microscopy images of eight parasite classes from a single unified dataset. All backbones are **frozen** — only a small linear head on top is trained (linear probing), keeping computational cost low for deployment in resource-constrained settings such as Brazil's SUS.

A second evaluation mode adds **multimodal late fusion**: the trained model's softmax output is combined with cosine-similarity scores retrieved from a Qdrant vector database indexed with class images and textual descriptions, using CLIP or SigLIP as the shared embedding backbone.

---

## Architecture

### Design Patterns

| Pattern | Where | Purpose |
|---|---|---|
| **Factory** | `src/models/factory.py` | Single entry point `ModelFactory.create(name, num_classes)` for all models |
| **Strategy** | `src/features/preprocessors.py` | Pluggable preprocessing; `HairRemovalFilter` for clinical images |
| **Template Method** | `src/models/nets/base.py` | `BaseModel` handles backbone freezing; subclasses define the head |

### Module Map

| Path | Responsibility |
|---|---|
| `src/run_experiment.py` | CLI: parse args → dataset → model → training loop |
| `src/run_test.py` | Load checkpoint → compute test metrics + confusion matrix |
| `src/run_test_multimodal.py` | Multimodal late-fusion evaluation (linear probe + VectorDB) |
| `src/models/factory.py` | `ModelFactory` registry |
| `src/models/trainer.py` | Training loop, early stopping (best val loss), curve export |
| `src/data/dataset.py` | `NTDDataset` — `__getitem__` returns `(pixel_values, label, path)` |
| `src/data/make_dataset.py` | Kaggle download, stratified 70/15/15 split |
| `src/features/preprocessors.py` | `HairRemovalFilter` (Black-Hat morphology + TELEA inpainting) |
| `src/multimodal/config.py` | Multimodal constants: paths, weights, allowed models |
| `src/multimodal/indexer.py` | Build Qdrant index from `vector_base/` images and texts |
| `src/multimodal/predictor.py` | `MultimodalPredictor` — late-fusion inference |
| `src/server/server.py` | FastAPI application with CORS |
| `src/api/router.py` | `/predict` REST endpoint (image upload) |
| `src/services/inference.py` | `get_model()` (LRU cache), `run_inference()` |
| `src/config/config.py` | Pydantic `BaseSettings`; `ALLOWED_MODELS` gates the API |
| `src/utils/logger.py` | `setup_logger()` — structured logging to file + console |
| `src/generate_convergence_plots.py` | Convergence plots from all `history.json` files |
| `src/generate_table_results.py` | Summary table across all evaluated models |

### Output Layout

```
output/
  results/{model_tag}/
    saved_model/best_{model_tag}.pth   # best checkpoint (val loss)
    history.json                        # per-epoch metrics
    loss_curve.pdf
    accuracy_curve.pdf
    log-train.log
    test_report/
      test_report.txt / .json
      test_predictions.csv
      confusion_matrix.pdf / .png
  results/multimodal_{model_tag}/      # late-fusion results
    test_report/  (same structure)
  multimodal_vector_db/                # Qdrant persistent storage
  figures/                             # EDA PDFs
logs/                                  # timestamped run logs
```

---

## Dataset

A single unified microscopy parasite dataset with **8 classes**, stratified-split 70 / 15 / 15 (train / val / test):

| Class | Train | Test |
|---|---|---|
| `microscopy_parasite_babesia` | 821 | 166 |
| `microscopy_parasite_leishmania` | 1 890 | 396 |
| `microscopy_parasite_leukocyte` | 963 | 197 |
| `microscopy_parasite_plasmodium` | 590 | 117 |
| `microscopy_parasite_rbcs` | 6 296 | 1 340 |
| `microscopy_parasite_toxoplasma` | 4 683 | 994 |
| `microscopy_parasite_trichomonad` | 7 093 | 1 511 |
| `microscopy_parasite_trypanosome` | 1 669 | 348 |

Source: [Parasite Dataset — Kaggle](https://www.kaggle.com/datasets/ahmedxc4/parasite-dataset)

---

## Model Registry

All models are registered in `src/models/factory.py`. Supported values for `--nets`:

| Model ID | Type | Embedding dim |
|---|---|---|
| `openai/clip-vit-base-patch16` | VLM | 512 |
| `openai/clip-vit-base-patch32` | VLM | 512 |
| `openai/clip-vit-large-patch14` | VLM | 768 |
| `google/siglip-base-patch16-224` | VLM | 768 |
| `google/vit-base-patch16-224` | ViT | 768 |
| `microsoft/resnet-50` | CNN | 2 048 |
| `google/efficientnet-b3` | CNN | 1 536 |

Only `openai/clip-vit-base-patch16` and `google/siglip-base-patch16-224` are supported for multimodal late fusion.

---

## Multimodal Late Fusion

`run_test_multimodal.py` combines three independent signals per image:

```
Final score = 0.5 × linear_probe_softmax
            + 0.3 × cosine_similarity_to_indexed_images (VectorDB)
            + 0.2 × cosine_similarity_to_indexed_texts  (VectorDB)
```

The Qdrant vector database is built from a knowledge base at `dataset/processed/Dataset-NTD-V1/vector_base/` with the following structure per class:

```
vector_base/
  {class_name}/
    IMAGEM/          ← reference images (indexed as image embeddings)
      img1.jpg
      ...
    informações.txt  ← textual class description (indexed as text embedding)
```

CLIP and SigLIP share the same embedding space for images and text, which makes cross-modal cosine search meaningful without any additional training.

---

## Setup

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/)
- CUDA 12.x (optional, CPU fallback available)
- Kaggle account with API token

### Install

```bash
git clone https://github.com/MarioCarvalhoBr/multimodal-ntd-classifier.git
cd multimodal-ntd-classifier
poetry install
```

### Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
KAGGLE_USERNAME="your_username"
KAGGLE_KEY="your_api_key"
PROCESSED_DATA_DIR="dataset/processed/Dataset-NTD-V1"
```

---

## Usage

### 1. Download and split the dataset

```bash
poetry run python src/data/make_dataset.py
```

Produces `dataset/processed/Dataset-NTD-V1/{train,val,test}/{class}/`.

### 2. Exploratory data analysis

```bash
poetry run python src/data/exploratory_analysis.py
```

Saves PDFs to `output/figures/`.

### 3. Train a model

```bash
poetry run python src/run_experiment.py \
  --classes microscopy_parasite_babesia microscopy_parasite_plasmodium \
            microscopy_parasite_trichomonad microscopy_parasite_leishmania \
            microscopy_parasite_rbcs microscopy_parasite_trypanosome \
            microscopy_parasite_leukocyte microscopy_parasite_toxoplasma \
  --nets openai/clip-vit-base-patch16 google/siglip-base-patch16-224 \
  --epochs 30 --batch_size 128
```

### 4. Evaluate on the test set (standard)

```bash
poetry run python src/run_test.py \
  --classes microscopy_parasite_babesia ... \
  --nets openai/clip-vit-base-patch16 google/siglip-base-patch16-224 \
  --batch_size 128
```

### 5. Evaluate with multimodal late fusion

Build the vector index on the first run, then reuse it:

```bash
# First run — builds the Qdrant index from vector_base/
poetry run python src/run_test_multimodal.py \
  --classes microscopy_parasite_babesia ... \
  --nets openai/clip-vit-base-patch16 google/siglip-base-patch16-224 \
  --build-index

# Subsequent runs (index already exists)
poetry run python src/run_test_multimodal.py \
  --classes microscopy_parasite_babesia ... \
  --nets openai/clip-vit-base-patch16

# Force index rebuild
poetry run python src/run_test_multimodal.py ... --build-index --force-reindex
```

Reports are saved to `output/results/multimodal_{model_tag}/test_report/`.

### 6. Aggregate results

```bash
poetry run python src/generate_convergence_plots.py
poetry run python src/generate_table_results.py
```

### 7. Shell scripts (GPU runs)

```bash
bash run_all_experiments_gpu_0.sh   # train + test all CNN/ViT models on GPU 0
bash run_one_experiment_gpu_0.sh    # single model on GPU 0
bash run_one_test_gpu_0.sh          # test only on GPU 0
bash run_one_test_multimodal_gpu_0.sh  # multimodal test on GPU 0
```

### 8. Backend API + Frontend

```bash
# Terminal 1 — API
poetry run uvicorn src.server.server:app --reload   # port 8000

# Terminal 2 — UI: Into the frontend/ directory and serve it
cd frontend/
python -m http.server 8080                          # serves frontend/
```

The `/predict` endpoint accepts an image upload and returns top-K class scores.

### Check GPU availability

```bash
poetry run python src/gpu_available.py
```

---

## Adding a New Model

1. Create `src/models/nets/your_model.py` extending `BaseModel`.
2. Register it in `src/models/factory.py` under its HuggingFace model ID.
3. Add the ID to `ALLOWED_MODELS` in `src/config/config.py` to expose it via the API.

---

## Citation

If you use this code in your research, please cite our work submitted to I CADTN:

Repository citation in BibTeX format:
```bibtex
@misc{carvalho2026ntdclassifier,
  author  = {Mário de Araújo Carvalho, Allison Oliveira Miranda, Celso Soares Costa, Wesley Nunes Gonçalves},
  title   = {Multimodal Vision-Language Classifier for Neglected Tropical Diseases},
  year    = {2026},
  url     = {https://github.com/MarioCarvalhoBr/multimodal-ntd-classifier}
}
```
Papper citation in BibTeX format:
```bibtex
@misc{carvalho2026papercadtn,
  author  = {Mário de Araújo Carvalho, Allison Oliveira Miranda, Celso Soares Costa, Wesley Nunes Gonçalves},
  title   = {Multimodal Artificial Intelligence for Aided Diagnosis of Neglected Tropical Diseases: A Comparative Study of Visual Architectures},
  year    = {2026},
  url     = {https://github.com/MarioCarvalhoBr/multimodal-ntd-classifier}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author

**Mário de Araújo Carvalho** — PhD student in Computer Science, specialist in Computer Vision and Deep Learning.  
[GitHub](https://github.com/MarioCarvalhoBr) · [LinkedIn](https://www.linkedin.com/in/mariodearaujocarvalho/)
