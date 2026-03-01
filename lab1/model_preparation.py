import os
import sys
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PREP_PATH = os.path.join(ROOT_DIR, "train", "data_preprocessed.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")


def main() -> int:
    if not os.path.exists(TRAIN_PREP_PATH):
        print("Ошибка: не найден файл с предобработанными данными train.")
        return 1

    df = pd.read_csv(TRAIN_PREP_PATH)

    target_col = "DAYTON_MW"

    if target_col not in df.columns:
        print("Ошибка: целевая переменная не найдена.")
        return 2

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = X.select_dtypes(include=["number"])

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Модель успешно обучена.")
    print(f"Модель сохранена в файл: {MODEL_PATH}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
