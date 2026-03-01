import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PREP_PATH = os.path.join(ROOT_DIR, "test", "data_preprocessed.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")
METRIC_FILE = os.path.join(ROOT_DIR, "test_metric.txt")
TARGET_COL = "DAYTON_MW"


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        print("Ошибка: не найден файл модели model.pkl. Сначала выполните model_preparation.py.")
        return 1

    if not os.path.exists(TEST_PREP_PATH):
        print("Ошибка: не найден файл с предобработанными тестовыми данными test/data_preprocessed.csv.")
        print("Сначала выполните data_preprocessing.py.")
        return 2

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(TEST_PREP_PATH)

    if TARGET_COL not in df.columns:
        print(f"Ошибка: целевая переменная '{TARGET_COL}' не найдена в тестовых данных.")
        return 3

    X_test = df.drop(columns=[TARGET_COL])
    X_test = X_test.select_dtypes(include=["number"])
    y_test = df[TARGET_COL]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mse ** 0.5

    print("=" * 50)
    print("Результаты тестирования модели на данных из папки test")
    print("=" * 50)
    print(f"Количество образцов: {len(y_test)}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")
    print("=" * 50)
    with open(METRIC_FILE, "w", encoding="utf-8") as f:
        f.write(f"{r2:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
