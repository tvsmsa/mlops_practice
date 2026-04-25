from app.model import MODEL_PATH, train_and_save_model


if __name__ == "__main__":
    artifact = train_and_save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training accuracy: {artifact['training_accuracy']:.4f}")
