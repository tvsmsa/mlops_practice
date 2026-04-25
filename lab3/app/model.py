from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.json"

FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TRAINING_DATA = [
    ("setosa", [5.1, 3.5, 1.4, 0.2]),
    ("setosa", [4.9, 3.0, 1.4, 0.2]),
    ("setosa", [4.7, 3.2, 1.3, 0.2]),
    ("setosa", [4.6, 3.1, 1.5, 0.2]),
    ("setosa", [5.0, 3.6, 1.4, 0.2]),
    ("setosa", [5.4, 3.9, 1.7, 0.4]),
    ("versicolor", [7.0, 3.2, 4.7, 1.4]),
    ("versicolor", [6.4, 3.2, 4.5, 1.5]),
    ("versicolor", [6.9, 3.1, 4.9, 1.5]),
    ("versicolor", [5.5, 2.3, 4.0, 1.3]),
    ("versicolor", [6.5, 2.8, 4.6, 1.5]),
    ("versicolor", [5.7, 2.8, 4.5, 1.3]),
    ("virginica", [6.3, 3.3, 6.0, 2.5]),
    ("virginica", [5.8, 2.7, 5.1, 1.9]),
    ("virginica", [7.1, 3.0, 5.9, 2.1]),
    ("virginica", [6.3, 2.9, 5.6, 1.8]),
    ("virginica", [6.5, 3.0, 5.8, 2.2]),
    ("virginica", [7.6, 3.0, 6.6, 2.1]),
]


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def train_and_save_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Train a nearest-centroid classifier and save it as a JSON artifact."""
    grouped: dict[str, list[list[float]]] = {}
    for class_name, features in TRAINING_DATA:
        grouped.setdefault(class_name, []).append(features)

    centroids = {}
    for class_name, rows in grouped.items():
        centroids[class_name] = [
            round(sum(row[index] for row in rows) / len(rows), 4)
            for index in range(len(FEATURE_NAMES))
        ]

    correct = 0
    for class_name, features in TRAINING_DATA:
        predicted = min(
            centroids,
            key=lambda candidate: _euclidean_distance(features, centroids[candidate]),
        )
        correct += int(predicted == class_name)

    artifact = {
        "model_type": "NearestCentroidClassifier",
        "feature_names": FEATURE_NAMES,
        "classes": sorted(centroids.keys()),
        "centroids": centroids,
        "training_accuracy": round(correct / len(TRAINING_DATA), 4),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the trained model. If the artifact is absent, train it once."""
    if not model_path.exists():
        return train_and_save_model(model_path)
    return json.loads(model_path.read_text(encoding="utf-8"))


def predict(features: list[float], artifact: dict[str, Any]) -> dict[str, Any]:
    """Return predicted Iris class and distance-based confidence scores."""
    centroids = artifact["centroids"]
    distances = {
        class_name: _euclidean_distance(features, centroid)
        for class_name, centroid in centroids.items()
    }
    predicted_class = min(distances, key=distances.get)

    inverse_distances = {
        class_name: 1 / (distance + 1e-9)
        for class_name, distance in distances.items()
    }
    total = sum(inverse_distances.values())

    return {
        "class_name": predicted_class,
        "confidence": round(inverse_distances[predicted_class] / total, 4),
        "scores": {
            class_name: round(inverse_distance / total, 4)
            for class_name, inverse_distance in inverse_distances.items()
        },
        "distances": {
            class_name: round(distance, 4)
            for class_name, distance in distances.items()
        },
    }

if __name__ == "__main__":
    artifact = train_and_save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training accuracy: {artifact['training_accuracy']:.4f}")