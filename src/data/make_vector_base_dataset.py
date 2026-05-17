#!/usr/bin/env python3
"""Create vector_base dataset from test split.

Moves 10% of each class from test/ into vector_base/{class}/IMAGEM/ using SEED=42,
and writes a resumo.txt with moved files and per-class counts.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

SEED = 42

DATASET_ROOT = Path(
    "/home/aroeira/Desktop/CARVALHO/doutorado/outros/multimodal-ntd-classifier/"
    "dataset/processed/Dataset-NTD-V1"
)
TEST_DIR = DATASET_ROOT / "test"
VECTOR_BASE_DIR = DATASET_ROOT / "vector_base"
SUMMARY_PATH = DATASET_ROOT / "resumo.txt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _pick_count(total: int) -> int:
    if total <= 0:
        return 0
    # Move at least 1 when there are images available.
    return max(1, int(total * 0.1))


def main() -> None:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    VECTOR_BASE_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    moved_records: list[tuple[str, str, str]] = []
    summary_rows: list[tuple[str, int, int]] = []

    for class_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        class_name = class_dir.name
        images = _list_images(class_dir)
        move_count = _pick_count(len(images))

        dest_dir = VECTOR_BASE_DIR / class_name / "IMAGEM"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if move_count > 0 and images:
            chosen = rng.sample(images, k=min(move_count, len(images)))
            for src in chosen:
                dest = dest_dir / src.name
                shutil.move(str(src), str(dest))
                moved_records.append((class_name, str(src), str(dest)))

        # Recount after moving
        remaining = len(_list_images(class_dir))
        moved = len(_list_images(dest_dir))
        summary_rows.append((class_name, remaining, moved))

    # Write summary
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("VECTOR_BASE DATASET SUMMARY\n")
        f.write(f"SEED: {SEED}\n\n")

        f.write("MOVED FILES\n")
        if moved_records:
            for class_name, src, dest in moved_records:
                f.write(f"- {class_name}: {src} -> {dest}\n")
        else:
            f.write("- (none)\n")

        f.write("\nCOUNTS BY CLASS\n")
        for class_name, remaining, moved in summary_rows:
            f.write(
                f"- {class_name}: test={remaining}, vector_base={moved}\n"
            )


if __name__ == "__main__":
    main()
