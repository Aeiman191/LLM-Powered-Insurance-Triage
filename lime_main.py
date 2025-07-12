from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer
import pandas as pd
import torch
import numpy as np
import uvicorn

# ==== FastAPI App ====
app = FastAPI()

# ==== Input Schema ====
class TextRequest(BaseModel):
    text: str

# ==== Load Model & Tokenizer ====
df = pd.read_csv("claims_final.csv", quotechar='"')
label_encoder = LabelEncoder()
df["SeverityEncoded"] = label_encoder.fit_transform(df["SeverityLabel"])
class_names = list(label_encoder.classes_)

model_path = "roberta_severity_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

print("✅ Model and tokenizer loaded.")

# ==== LIME-Compatible Prediction Function ====
def predict_proba(texts):
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**encodings)
        probs = torch.nn.functional.softmax(output.logits, dim=1)
    return probs.detach().numpy()

# ==== Prediction Endpoint ====
@app.post("/predict/")
def predict_severity(request: TextRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # LIME explanation
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=5
    )

    explanation_data = [
        {"word": word, "importance": float(score)}
        for word, score in explanation.as_list()
    ]

    return {
        "predicted_label": predicted_label,
        "lime_explanation": explanation_data
    }


