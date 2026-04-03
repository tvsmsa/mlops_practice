"""
Скачивание датасета с Kaggle.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = RAW_DIR / "dataset.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Скачивание датасета 'vitthalmadane/ts-temp-1'...")

    dataset_path = kagglehub.dataset_download("vitthalmadane/ts-temp-1")
    dataset_path_obj = Path(dataset_path)
    csv_files = list(dataset_path_obj.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV файл не найден в {dataset_path}")

    shutil.copy(csv_files[0], OUTPUT)
    print(f"Сохранено: {OUTPUT} ({OUTPUT.stat().st_size} байт)")


if __name__ == "__main__":
    main()