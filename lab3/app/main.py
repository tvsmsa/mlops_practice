from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import FEATURE_NAMES, load_model, predict

app = FastAPI(
    title="Lab3 Iris ML Microservice",
    description="ML microservice that predicts Iris flower species by four numeric features.",
    version="1.0.0",
)

ARTIFACT = load_model()


class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])

    def as_features(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "lab3-iris-ml-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model_type": ARTIFACT["model_type"],
        "features": FEATURE_NAMES,
        "classes": ARTIFACT["classes"],
        "training_accuracy": ARTIFACT["training_accuracy"],
        "centroids": ARTIFACT["centroids"],
    }


@app.post("/predict")
def make_prediction(payload: IrisRequest) -> dict[str, object]:
    result = predict(payload.as_features(), ARTIFACT)
    return {
        "input": payload.model_dump(),
        "prediction": result,
    }
