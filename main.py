from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd

# Load dataset and encode labels
df = pd.read_csv("claims_final.csv", quotechar='"')
label_encoder = LabelEncoder()
df["SeverityEncoded"] = label_encoder.fit_transform(df["SeverityLabel"])

# Load model and tokenizer
model_path = "roberta_severity_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

# Create FastAPI app
app = FastAPI()

# Define input data model
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_severity(input: TextInput):
    text = input.text

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_idx = torch.argmax(logits, dim=1).item()

    # Decode to severity label
    predicted_label = label_encoder.inverse_transform([prediction_idx])[0]

    return {
        "predicted_severity": predicted_label
    }
