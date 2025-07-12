from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import numpy as np
import shap

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

# ==== SHAP-Compatible Prediction Function ====
def predict_proba_shap(texts):
    str_texts = [str(t) for t in texts]
    encodings = tokenizer(str_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**encodings)
        probs = torch.nn.functional.softmax(output.logits, dim=1)
    return probs.detach().numpy()

# ==== SHAP Explanation Wrapper ====
def shap_explanation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # SHAP explainer
    explainer = shap.Explainer(predict_proba_shap, tokenizer)
    shap_values = explainer([text])
    explanation = shap_values[0]

    class_index = prediction
    words = explanation.data
    values = explanation.values[:, class_index]
    base_value = explanation.base_values[class_index]

    def get_prob_from_shap(base, contribs):
        logits = base + contribs
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    approx_prob = get_prob_from_shap(base_value, values.sum())

    def clean_token(token):
        return token.replace("Ġ", "").replace("Ļ", "").strip()

    top_words = sorted(zip(words, values), key=lambda x: abs(x[1]), reverse=True)[:5]
    readable = []
    for w, v in top_words:
        clean_w = clean_token(w)
        if clean_w and clean_w.isalpha():
            readable.append({"word": clean_w, "impact": float(v)})

    return predicted_label, readable, float(approx_prob)

# ==== FastAPI Endpoint ====
@app.post("/predict/")
def predict_severity(request: TextRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    predicted_label, shap_explanation_data, probability = shap_explanation(text)

    return {
        "predicted_label": predicted_label,
        "approximate_probability": f"{probability * 100:.2f}%",
        "shap_explanation": shap_explanation_data
    }
