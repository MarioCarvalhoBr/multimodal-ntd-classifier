"""
Multimodal test script — Late Fusion (I CADTN)

Fusion weights:
  0.4 → fine-tuned linear probe (trained .pth checkpoint)
  0.3 → cosine similarity to indexed images in VectorDB (backbone embedding)
  0.3 → cosine similarity to indexed text descriptions in VectorDB (backbone embedding)

Allowed models: google/siglip-base-patch16-224 | openai/clip-vit-base-patch16

Example:
    poetry run python src/run_test_multimodal.py \\
        --classes microscopy_parasite_babesia microscopy_parasite_leishmania \\
        --nets openai/clip-vit-base-patch16 \\
        --build-index

    # Force rebuild of the vector index:
    poetry run python src/run_test_multimodal.py ... --build-index --force-reindex
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NTDDataset
from multimodal.config import ALLOWED_MODELS, PESOS
from multimodal.indexer import build_index
from multimodal.predictor import MultimodalPredictor
from utils.logger import logger, log_file


# ---------------------------------------------------------------------------
# Reporting helpers (mirrors trainer.py style)
# ---------------------------------------------------------------------------

def _save_report(
    all_preds: list,
    all_labels: list,
    all_paths: list,
    classes: list,
    model_name: str,
) -> None:
    model_tag = model_name.replace("/", "_")
    output_dir = Path(f"output/results/multimodal_{model_tag}/test_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average="macro")

    logger.info("\n" + "=" * 60)
    logger.info("MÉTRICAS FINAIS — FUSÃO MULTIMODAL (TEST SET)")
    logger.info("=" * 60)
    logger.info(f"Acurácia: {test_acc:.4f} | F1-Macro: {test_f1:.4f}\n")
    logger.info(classification_report(all_labels, all_preds, target_names=classes))

    with open(output_dir / "test_report.txt", "w") as f:
        f.write("MÉTRICAS FINAIS — FUSÃO MULTIMODAL (TEST SET)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Acurácia: {test_acc:.4f} | F1-Macro: {test_f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=classes))

    with open(output_dir / "test_report.json", "w") as f:
        json.dump(
            {
                "accuracy":  test_acc,
                "f1_macro":  test_f1,
                "fusion_weights": PESOS,
                "classification_report": classification_report(
                    all_labels, all_preds, target_names=classes, output_dict=True
                ),
            },
            f,
            indent=4,
        )

    # Predictions CSV
    pd.DataFrame(
        {
            "image_path":    all_paths,
            "true_label_id": all_labels,
            "pred_label_id": all_preds,
            "true_class":    [classes[i] for i in all_labels],
            "pred_class":    [classes[i] for i in all_preds],
            "is_correct":    [t == p for t, p in zip(all_labels, all_preds)],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    short_names = [n.replace("microscopy_parasite_", "").capitalize() for n in classes]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm, annot=False, cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
        cbar=True, linewidths=1, linecolor="white", square=True,
    )

    limite_cor = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            cor = "white" if val > limite_cor else "black"
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha="center", va="center", fontsize=13, weight="bold", color=cor)

    plt.title(f"Matriz de Confusão Multimodal — {model_tag}", fontsize=16, pad=20, weight="bold")
    plt.ylabel("Classe Verdadeira", fontsize=14, weight="bold")
    plt.xlabel("Classe Prevista",   fontsize=14, weight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"[+] Resultados salvos em: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliação multimodal com late fusion — I CADTN"
    )
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help="Classes a avaliar (as mesmas usadas no treinamento)",
    )
    parser.add_argument(
        "--nets", nargs="+", required=True,
        help=f"Modelos VLM permitidos: {ALLOWED_MODELS}",
    )
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=0)
    parser.add_argument(
        "--build-index", action="store_true",
        help="Constrói o índice vetorial (vector_base → Qdrant) antes do teste",
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Reconstrói o índice mesmo que já exista",
    )
    args = parser.parse_args()

    # Validate model names
    for net in args.nets:
        if net not in ALLOWED_MODELS:
            logger.info(
                f"❌ Modelo '{net}' não é permitido no modo multimodal.\n"
                f"   Permitidos: {ALLOWED_MODELS}"
            )
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[+] Dispositivo: {device}")

    test_dir = Path("dataset/processed/Dataset-NTD-V1/test")
    if not test_dir.exists():
        logger.info(f"❌ Pasta de teste não encontrada: {test_dir}")
        return

    for model_name in args.nets:
        logger.info("\n" + "=" * 60)
        logger.info(f"[*] TESTE MULTIMODAL: {model_name}")

        # --- Indexing phase (optional) ---
        if args.build_index:
            build_index(model_name, force=args.force_reindex)

        # --- Build predictor (loads .pth + opens Qdrant) ---
        model_tag = model_name.replace("/", "_")
        ckpt = Path(f"output/results/{model_tag}/saved_model/best_{model_tag}.pth")
        if not ckpt.exists():
            logger.info(
                f"⚠️  Checkpoint não encontrado: {ckpt}. "
                "Treine o modelo primeiro com run_experiment.py."
            )
            continue

        predictor = MultimodalPredictor(
            model_name=model_name,
            num_classes=len(args.classes),
            classes=args.classes,
            device=str(device),
        )

        # --- Dataset uses the processor from the predictor ---
        test_dataset = NTDDataset(
            test_dir,
            predictor.processor,
            args.classes,
            custom_preprocessor=None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        # --- Inference loop ---
        all_preds:  list[int] = []
        all_labels: list[int] = []
        all_paths:  list[str] = []

        for batch in tqdm(test_loader, desc=f"Multimodal [{model_tag}]"):
            pixel_values = batch[0].to(device)
            labels       = batch[1].tolist()
            paths        = list(batch[2]) if len(batch) > 2 else [""] * len(labels)

            preds = predictor.predict_batch(pixel_values)

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_paths.extend(paths)

        predictor.close()

        # --- Reports ---
        _save_report(all_preds, all_labels, all_paths, args.classes, model_name)

        log_out = Path(f"output/results/multimodal_{model_tag}")
        log_out.mkdir(parents=True, exist_ok=True)
        if log_file:
            shutil.copy2(log_file, log_out / "log-test-multimodal.log")
            logger.info(f"[+] Log copiado para: {log_out / 'log-test-multimodal.log'}")

        logger.info(f"[*] Avaliação multimodal concluída para: {model_name}")


if __name__ == "__main__":
    main()
