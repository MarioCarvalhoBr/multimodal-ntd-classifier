#!/usr/bin/env python3
"""Create vector_base dataset from train split.

Copies 100% of each class from train/ into vector_base/{class}/IMAGEM/ and writes
a resumo.txt with per-class counts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

DATASET_ROOT = Path(
    "/home/aroeira/Desktop/CARVALHO/doutorado/outros/multimodal-ntd-classifier/"
    "dataset/processed/Dataset-NTD-V1"
)
TRAIN_DIR = DATASET_ROOT / "train"
VECTOR_BASE_DIR = DATASET_ROOT / "vector_base"
SUMMARY_PATH = DATASET_ROOT / "resumo.txt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

    VECTOR_BASE_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[str, int, int]] = []

    for class_dir in sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir()):
        class_name = class_dir.name
        images = _list_images(class_dir)

        dest_dir = VECTOR_BASE_DIR / class_name / "IMAGEM"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src in images:
            dest = dest_dir / src.name
            shutil.copy2(str(src), str(dest))

        train_count = len(images)
        vector_count = len(_list_images(dest_dir))
        summary_rows.append((class_name, train_count, vector_count))

    # Write summary
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("VECTOR_BASE DATASET SUMMARY\n")
        f.write("\nCOUNTS BY CLASS\n")
        for class_name, train_count, vector_count in summary_rows:
            f.write(
                f"- {class_name}: train={train_count}, vector_base={vector_count}\n"
            )


if __name__ == "__main__":
    main()
