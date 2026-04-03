"""
Обучение модели на train.csv.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(PROC / "train.csv")

    target_col = "DAYTON_MW"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = LinearRegression()
    model.fit(X, y)

    path = MODELS / "model.joblib"
    joblib.dump(model, path)

    print(f"Модель сохранена: {path}")
    print("Признаки модели:", list(X.columns))


if __name__ == "__main__":
    main()