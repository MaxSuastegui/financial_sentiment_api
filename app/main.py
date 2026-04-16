import os
import json
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "artifacts/model"

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        "No se encontró el modelo entrenado. "
        "Primero ejecuta: python training/train_model.py"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(os.path.join(MODEL_DIR, "label_mapping.json"), "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

id2label = {int(k): v for k, v in label_mapping["id2label"].items()}

app = FastAPI(title="Financial Sentiment API")


class InputData(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_id].item()

    return {
        "prediction_id": pred_id,
        "prediction_label": id2label[pred_id],
        "probability": float(pred_prob)
    }