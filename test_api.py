import requests

url = "http://127.0.0.1:8001/predict"

examples = [
    {"text": "The company reported strong quarterly earnings and rising revenue."},
    {"text": "The firm announced major losses and declining market confidence."},
    {"text": "The board maintained its outlook for the rest of the year."}
]

for example in examples:
    response = requests.post(url, json=example)
    print("Texto:", example["text"])
    print("Respuesta:", response.json())
    print("-" * 50)
