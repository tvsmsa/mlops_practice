"""
Скачивание датасета TS-Temp-1 с Kaggle.
Этап конвейера: получение сырых данных.
"""
from __future__ import annotations

import kagglehub
import shutil
from pathlib import Path

CURRENT_DIR = Path.cwd()
RAW_DIR = CURRENT_DIR / "data" / "raw"
OUTPUT = RAW_DIR / "dataset.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание датасета 'vitthalmadane/ts-temp-1'...")
    
    dataset_path = kagglehub.dataset_download("vitthalmadane/ts-temp-1")
    dataset_path_obj = Path(dataset_path)
    csv_files = list(dataset_path_obj.glob("*.csv"))
    
    if csv_files:
        print(f"Копирование {csv_files[0].name} в {OUTPUT}")
        shutil.copy(csv_files[0], OUTPUT)
        print(f"Сохранено: {OUTPUT} ({OUTPUT.stat().st_size} байт)")
    else:
        raise FileNotFoundError(f"CSV файл не найден в {dataset_path}")


if __name__ == "__main__":
    main()