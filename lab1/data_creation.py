import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def download_dataset():
    path = kagglehub.dataset_download("vitthalmadane/ts-temp-1")
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            source = os.path.join(path, file_name)
            destination = os.path.join('lab1', 'dataset.csv')
            shutil.copy(source, destination)
            print(f"Файл {file_name} скопирован в {destination}")
            break

def split_dataset():
    df = pd.read_csv("lab1/dataset.csv")
    os.makedirs("lab1/train", exist_ok=True)
    os.makedirs("lab1/test", exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_path = f'lab1/train/data.csv'
    test_path = f'lab1/test/data.csv'
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

if __name__ == "__main__":
    download_dataset()
    split_dataset()