"""
Оценка модели на тестовых данных.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def main() -> None:
    model = joblib.load(MODELS / "model.joblib")
    test = pd.read_csv(PROC / "test.csv")

    y_true = test["DAYTON_MW"]
    X = test.drop(columns=["DAYTON_MW"])

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    print("=== Метрики на тесте ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Model test R2 is: {r2:.4f}")


if __name__ == "__main__":
    main()