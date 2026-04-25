# Lab3: ML Microservice

В этом подкаталоге реализован простой микросервис машинного обучения на FastAPI.

Сервис использует простую модель классификации Iris. Пользователь отправляет четыре числовых признака цветка, а микросервис возвращает предсказанный класс ириса.

## Что реализовано

- Python-код модели: `app/model.py`
- Python-код микросервиса: `app/main.py`
- Скрипт обучения модели: `app/train_model.py`
- Сохранённая модель: `app/model.json`
- REST API на FastAPI

## Локальный запуск

```powershell
cd E:\mlops_practice\lab3

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

pip install --upgrade -r requirements.txt

python -m app.train_model

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000