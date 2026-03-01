import os
import sys
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(ROOT_DIR, "train", "data.csv")
TEST_PATH = os.path.join(ROOT_DIR, "test", "data.csv")

OUT_TRAIN_PATH = os.path.join(ROOT_DIR, "train", "data_preprocessed.csv")
OUT_TEST_PATH = os.path.join(ROOT_DIR, "test", "data_preprocessed.csv")


def expand_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    parsed = pd.to_datetime(df[col], errors="coerce")
    if parsed.isna().mean() > 0.5:
        return df
    df = df.copy()
    df[col] = parsed
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main() -> int:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df = expand_datetime_features(train_df, "Datetime")
    test_df = expand_datetime_features(test_df, "Datetime")
    target_col = "DAYTON_MW" if "DAYTON_MW" in train_df.columns else None

    numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = scaler.transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[feature_cols] = train_out[feature_cols].astype(float)
    test_out[feature_cols] = test_out[feature_cols].astype(float)
    train_out.loc[:, feature_cols] = train_scaled
    test_out.loc[:, feature_cols] = test_scaled

    os.makedirs(os.path.dirname(OUT_TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TEST_PATH), exist_ok=True)

    train_out.to_csv(OUT_TRAIN_PATH, index=False)
    test_out.to_csv(OUT_TEST_PATH, index=False)

    print(f"Предобработка завершена. Масштабированные столбцы: {feature_cols}")
    print(f"Файл обучающей выборки сохранён: {OUT_TRAIN_PATH}")
    print(f"Файл тестовой выборки сохранён: {OUT_TEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
