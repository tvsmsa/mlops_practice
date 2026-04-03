"""
Предобработка данных для регрессии DAYTON_MW.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "dataset.csv"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(RAW)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["day"] = df["Datetime"].dt.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["lag_1"] = df["DAYTON_MW"].shift(1)
    df["lag_24"] = df["DAYTON_MW"].shift(24)
    df["lag_168"] = df["DAYTON_MW"].shift(168)

    df["rolling_mean_24"] = df["DAYTON_MW"].shift(1).rolling(24).mean()
    df["rolling_mean_168"] = df["DAYTON_MW"].shift(1).rolling(168).mean()

    df = df.dropna().reset_index(drop=True)

    drop_cols = [c for c in ["Datetime", "Datetime1", "target"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Split по времени
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_path = PROC / "train.csv"
    test_path = PROC / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {train_path} ({len(train_df)} строк)")
    print(f"Test:  {test_path} ({len(test_df)} строк)")
    print("Колонки train:", list(train_df.columns))


if __name__ == "__main__":
    main()