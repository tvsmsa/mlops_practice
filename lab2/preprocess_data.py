"""
Предобработка: признаки, разбиение train/test, сохранение датасетов и scaler.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "dataset.csv"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(RAW)
    # ожидаем колонки species + числовые признаки
    target_col = "species" if "species" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df[target_col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_s, columns=feature_cols)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test_s, columns=feature_cols)
    test_df["target"] = y_test

    train_path = PROC / "train.csv"
    test_path = PROC / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, PROC / "scaler.joblib")
    joblib.dump(feature_cols, PROC / "feature_columns.joblib")

    print(f"Train: {train_path} ({len(train_df)} строк)")
    print(f"Test:  {test_path} ({len(test_df)} строк)")
    print(f"Признаки: {feature_cols}")


if __name__ == "__main__":
    main()