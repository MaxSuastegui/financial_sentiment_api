# Financial Sentiment API

## Descripción

Este proyecto consiste en la implementación de un servicio local utilizando FastAPI para realizar análisis de sentimientos sobre texto financiero.

Se utiliza un modelo de lenguaje basado en transformers (`DistilBERT`) para clasificar textos en tres categorías:

* positive
* negative
* neutral

El sistema permite recibir texto mediante un endpoint y devolver una predicción junto con su probabilidad.

---

## Dataset

Se utilizó el dataset:

`sbhatti/financial-sentiment-analysis`

Columnas principales:

* **Sentence**: texto de entrada
* **Sentiment**: etiqueta objetivo

---

## Tecnologías utilizadas

* Python
* FastAPI
* Hugging Face Transformers
* PyTorch
* Pandas
* Scikit-learn
* KaggleHub

---

## Estructura del proyecto

```
financial_sentiment_api/
│
├── app/
│   └── main.py
├── training/
│   └── train_model.py
├── artifacts/
│   └── model/
├── test_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Instalación

### 1. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Entrenamiento del modelo

```bash
python training/train_model.py
```

Este script:

* descarga el dataset desde KaggleHub
* procesa los datos
* entrena un modelo DistilBERT
* guarda el modelo en `artifacts/model/`

---

## Ejecución de la API

```bash
uvicorn app.main:app --reload --port 8001
```

Abrir en navegador:

```
http://127.0.0.1:8001/docs
```

---

## Uso del endpoint

### POST `/predict`

Ejemplo de request:

```json
{
  "text": "The company reported strong quarterly earnings and rising revenue."
}
```

Ejemplo de respuesta:

```json
{
  "prediction_id": 2,
  "prediction_label": "positive",
  "probability": 0.99
}
```

---

## Pruebas

También se puede probar mediante script:

```bash
python test_api.py
```

---

## Notas

* El proyecto se ejecuta completamente de forma local
* Se utilizó un transformer pequeño para mantener eficiencia
* No se utilizaron servicios en la nube (Azure no requerido en esta actividad)

---

## Autores

Maximiliano García Suástegui
Carlos Gómez San Pedro
Arath Mendivil Mora

Proyecto académico – Curso de Cloud Computing (2026)
