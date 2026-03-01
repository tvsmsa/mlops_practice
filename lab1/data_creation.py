import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, "dataset.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
TRAIN_PATH = os.path.join(TRAIN_DIR, "data.csv")
TEST_PATH = os.path.join(TEST_DIR, "data.csv")


def download_dataset():
    path = kagglehub.dataset_download("vitthalmadane/ts-temp-1")
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            source = os.path.join(path, file_name)
            shutil.copy(source, DATASET_PATH)
            print(f"Файл {file_name} скопирован в {DATASET_PATH}")
            break


def split_dataset():
    df = pd.read_csv(DATASET_PATH)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

if __name__ == "__main__":
    download_dataset()
    split_dataset()