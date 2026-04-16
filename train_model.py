import os
import json
import pandas as pd
import kagglehub
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset


MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "artifacts/model"


class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def main():
    print("Descargando dataset desde Kaggle...")
    path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
    csv_path = os.path.join(path, "data.csv")

    print(f"Dataset descargado en: {csv_path}")

    df = pd.read_csv(csv_path)

    print("Columnas reales del dataset:")
    print(df.columns.tolist())
    print("\nPrimeras filas:")
    print(df.head())

    df.columns = df.columns.str.strip().str.lower()

    df = df[["sentence", "sentiment"]].dropna()

    X = df["sentence"].tolist()
    y = df["sentiment"].tolist()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FinancialSentimentDataset(X_train, y_train, tokenizer)
    eval_dataset = FinancialSentimentDataset(X_test, y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )

    training_args = TrainingArguments(
        output_dir="artifacts/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="artifacts/logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    print("Entrenando modelo...")
    trainer.train()

    print(f"Guardando modelo en {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"id2label": id2label, "label2id": label2id},
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Entrenamiento terminado.")


if __name__ == "__main__":
    main()
