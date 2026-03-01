set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[1/4] data_creation.py"
python data_creation.py

echo "[2/4] data_preprocessing.py"
python data_preprocessing.py

echo "[3/4] model_preparation.py"
python model_preparation.py

echo "[4/4] model_testing.py"
python model_testing.py

if [ ! -f "test_metric.txt" ]; then
  echo "Ошибка: файл метрики test_metric.txt не создан." >&2
  exit 1
fi
METRIC=$(cat test_metric.txt | tr -d '[:space:]')
echo "Model test R² is: $METRIC"
