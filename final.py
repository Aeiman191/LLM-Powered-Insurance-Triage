from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO, BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import numpy as np



app = FastAPI()

# Use a small model for local development
model_name = "gpt2"  # You can change to "tiiuae/falcon-rw-1b" if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to generate realistic claim description
def generate_detailed_description(claim: str) -> str:
    prompt = (
        f"Convert the following injury keywords into a realistic claim statement. "
        f"The sentence should sound like something a person would say when reporting the incident.\n\n"
        f"Keywords: {claim.lower().capitalize()}\n"
        f"Statement:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.replace(prompt, "").strip()

# Function to determine severity label from IncurredQuantile
def determine_severity(row):
    if row['IncurredQuantile'] == 'HighCost':
        return 'High'
    elif row['IncurredQuantile'] == 'LowCost':
        return 'Low'
    else:
        return 'Medium'
    

# --- Template mappings ---
gender_map = {'M': 'male', 'F': 'female'}
marital_map = {'S': 'single', 'M': 'married', 'U': 'unmarried'}
# === Mapping dictionaries ===
gender_map = {'M': 'male', 'F': 'female'}
marital_map = {'S': 'single', 'M': 'married', 'U': 'unmarried'}
work_type_map = {'F': 'full-time', 'P': 'part-time'}

# === Templates for each column (diverse paraphrasing) ===
templates = {
    "Age": [
        "I am {Age} years old.",
        "I'm currently {Age}.",
        "My age is {Age}.",
        "At present, I am {Age} years of age.",
        "I’ve reached the age of {Age}.",
        "I’m aged {Age}.",
        "Currently, I am {Age}.",
        "I was born {Age} years ago.",
        "I fall in the {Age}-year age group.",
        "I’ve lived for {Age} years now."
    ],
    "Gender": [
        "I am a {Gender}.",
        "I identify as {Gender}.",
        "Gender: {Gender}.",
        "My gender is {Gender}.",
        "As a {Gender}, I face certain challenges.",
        "Being {Gender} has influenced my experience.",
        "I live my life as a {Gender}.",
        "I was registered as {Gender} at birth.",
        "Socially, I identify as {Gender}.",
        "Biologically, I am {Gender}."
    ],
    "MaritalStatus": [
        "I am {MaritalStatus}.",
        "I'm currently {MaritalStatus}.",
        "Marital status: {MaritalStatus}.",
        "I live as a {MaritalStatus} person.",
        "I consider myself to be {MaritalStatus}.",
        "Currently, I’m {MaritalStatus} in terms of relationship status.",
        "In terms of my personal life, I'm {MaritalStatus}.",
        "I have a {MaritalStatus} lifestyle.",
        "Relationship-wise, I’m {MaritalStatus}.",
        "I’m leading a {MaritalStatus} life."
    ],
    "DateOfAccident": [
        "I had an accident on {DateOfAccident}.",
        "The accident occurred on {DateOfAccident}.",
        "It happened on {DateOfAccident}.",
        "The incident took place on {DateOfAccident}.",
        "The mishap happened on {DateOfAccident}.",
        "That unfortunate day was {DateOfAccident}.",
        "I remember it clearly—it was {DateOfAccident} when the accident occurred.",
        "The event happened on {DateOfAccident}.",
        "I met with the accident on {DateOfAccident}.",
        "The injury happened on {DateOfAccident}."
    ],
    "ReportedDay": [
        "It was reported on {ReportedDay}.",
        "I made the report on {ReportedDay}.",
        "The report was filed on {ReportedDay}.",
        "Reported day: {ReportedDay}.",
        "I officially reported it on {ReportedDay}.",
        "The day I submitted the report was {ReportedDay}.",
        "I remember reporting it on {ReportedDay}.",
        "I contacted the authorities on {ReportedDay}.",
        "I filed the claim on {ReportedDay}.",
        "The issue was recorded on {ReportedDay}."
    ],
    "PartTimeFullTime": [
        "I work {PartTimeFullTime}.",
        "My job is {PartTimeFullTime}.",
        "Employment type: {PartTimeFullTime}.",
        "I am employed {PartTimeFullTime}.",
        "I currently have a {PartTimeFullTime} job.",
        "My role requires me to work in a {PartTimeFullTime} capacity.",
        "I’m engaged in {PartTimeFullTime} employment.",
        "I am a {PartTimeFullTime} employee.",
        "I work on a {PartTimeFullTime} basis.",
        "My job type is {PartTimeFullTime}."
    ],
    "HoursWorkedPerWeek": [
        "I usually work {HoursWorkedPerWeek} hours per week.",
        "My weekly working hours are {HoursWorkedPerWeek}.",
        "I work {HoursWorkedPerWeek} hours each week.",
        "Average weekly hours: {HoursWorkedPerWeek}.",
        "In a typical week, I put in about {HoursWorkedPerWeek} hours.",
        "I dedicate {HoursWorkedPerWeek} hours weekly to my job.",
        "Every week, I spend around {HoursWorkedPerWeek} hours working.",
        "My workload per week totals {HoursWorkedPerWeek} hours.",
        "On average, I log {HoursWorkedPerWeek} work hours weekly.",
        "I’m scheduled for {HoursWorkedPerWeek} hours every week."
    ],
    "DependentChildren": [
        "I have {DependentChildren} dependent children.",
        "Number of dependent children: {DependentChildren}.",
        "There are {DependentChildren} children depending on me.",
        "Dependents (children): {DependentChildren}.",
        "I take care of {DependentChildren} kids.",
        "I’m responsible for {DependentChildren} dependent children.",
        "My household includes {DependentChildren} dependent children.",
        "I support {DependentChildren} children financially.",
        "I raise {DependentChildren} children.",
        "I provide for {DependentChildren} kids."
    ],
    "DependentsOther": [
        "I support {DependentsOther} other dependents.",
        "Dependents (others): {DependentsOther}.",
        "There are {DependentsOther} other people who depend on me.",
        "I look after {DependentsOther} non-child dependents.",
        "I financially support {DependentsOther} other individuals.",
        "Aside from children, I care for {DependentsOther} others.",
        "I have {DependentsOther} additional dependents under my care.",
        "I’m responsible for {DependentsOther} non-minor dependents.",
        "There are {DependentsOther} adults I help support.",
        "I assist {DependentsOther} others with living expenses."
    ],
    "WeeklyRate": [
        "My weekly rate is {WeeklyRate}.",
        "I earn {WeeklyRate} per week.",
        "Weekly wage: {WeeklyRate}.",
        "Each week, I get paid {WeeklyRate}.",
        "My income per week amounts to {WeeklyRate}.",
        "The amount I make weekly is {WeeklyRate}.",
        "I receive {WeeklyRate} weekly as part of my job.",
        "My weekly earnings total {WeeklyRate}.",
        "I get compensated {WeeklyRate} each week.",
        "The company pays me {WeeklyRate} weekly."
    ]
}


severity_templates = {
    "Low": [
        "It was a minor incident and I was back to work shortly.",
        "No major injuries were sustained, and I resumed duties soon.",
        "The injury wasn’t too serious, and I recovered quickly.",
        "I didn’t need much medical help and returned to work fast.",
        "I managed with basic care and had minimal downtime.",
        "The discomfort was manageable, and I stayed productive.",
        "Luckily, it wasn't severe and didn’t affect my work much.",
        "Only light first aid was needed, and I was fine afterward.",
        "There was no lasting impact, and I got better quickly.",
        "I was able to handle the injury with ease and return to work."
    ],
    "Medium": [
        "The injury required some medical attention and time off.",
        "I had to rest for a few days and get treated for the injury.",
        "It wasn’t critical, but it did affect my work temporarily.",
        "There was moderate pain and disruption to my routine.",
        "I missed a few shifts and received treatment.",
        "I had to take a break from work to recover properly.",
        "The incident led to a brief period of downtime.",
        "My workflow was impacted, but I managed to recover soon.",
        "I had to visit a doctor and follow up with rest.",
        "It was concerning enough to pause work temporarily."
    ],
    "High": [
        "The injury was severe and required immediate medical care.",
        "I was hospitalized and needed significant recovery time.",
        "It greatly impacted my ability to work and daily life.",
        "There was substantial pain and long-term treatment involved.",
        "I had to stop working for an extended period.",
        "The situation was critical and required surgery.",
        "Rehabilitation and time away from work were necessary.",
        "I experienced long-term discomfort from the injury.",
        "The accident disrupted my life and career significantly.",
        "It was a serious incident with lasting consequences."
    ]
}


# --- Function to generate InputText ---
import pickle

def generate_input_text(row):
    gender = gender_map.get(row["Gender"], "unknown")
    marital = marital_map.get(row["MaritalStatus"], "unknown")
    work_type = work_type_map.get(row["PartTimeFullTime"], "unknown")

    # Prepare values with defaults
    values = {
        "Age": int(row["Age"]) if pd.notnull(row["Age"]) else "unknown",
        "Gender": gender,
        "MaritalStatus": marital,
        "DateOfAccident": row["DateOfAccident"] if pd.notnull(row["DateOfAccident"]) else "unknown",
        "ReportedDay": row["ReportedDay"] if pd.notnull(row["ReportedDay"]) else "unknown",
        "PartTimeFullTime": work_type,
        "HoursWorkedPerWeek": int(row["HoursWorkedPerWeek"]) if pd.notnull(row["HoursWorkedPerWeek"]) else "unknown",
        "DependentChildren": int(row["DependentChildren"]) if pd.notnull(row["DependentChildren"]) else "unknown",
        "DependentsOther": int(row["DependentsOther"]) if pd.notnull(row["DependentsOther"]) else "unknown",
        "WeeklyRate": int(row["WeeklyRate"]) if pd.notnull(row["WeeklyRate"]) and row["WeeklyRate"] > 0 else "not specified"
    }

    # Randomly choose how many and which fields to include
    available_fields = [k for k, v in values.items() if v != "unknown"]
    selected_fields = random.sample(available_fields, k=random.randint(1, len(available_fields)))

    parts = []
    for field in selected_fields:
        if field in templates:
            template = random.choice(templates[field])
            parts.append(template.format(**values))

    # Always include the detailed description and severity
    if pd.notnull(row.get("DetailedClaimDescription")):
        parts.append(f"Description: {row['DetailedClaimDescription']}")

    if pd.notnull(row.get("SeverityLabel")):
        parts.append(f"The severity level is {row['SeverityLabel']}.")

    return " ".join(parts)


# Global for processed data
processed_df = pd.DataFrame()



@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    global processed_df

    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or XLSX file.")

    contents = await file.read()

    # Read the file into a DataFrame
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
        else:  # .xlsx
            df = pd.read_excel(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Check required columns
    if "ClaimDescription" not in df.columns or "Incurred" not in df.columns:
        raise HTTPException(status_code=400, detail="File must contain 'ClaimDescription' and 'Incurred' columns.")

    # Take a sample for transformation
    subset_df = df.head(50).copy()

    # Generate claim descriptions
    subset_df["DetailedClaimDescription"] = subset_df["ClaimDescription"].apply(generate_detailed_description)

    # Quantile binning on Incurred
    try:
        subset_df["IncurredQuantile"] = pd.qcut(subset_df["Incurred"], q=3, labels=["LowCost", "MediumCost", "HighCost"])
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Error computing Incurred quantiles: {ve}")

    # Assign severity label
    subset_df["SeverityLabel"] = subset_df.apply(determine_severity, axis=1)

    # Generate InputText
    subset_df["InputText"] = subset_df.apply(generate_input_text, axis=1)

    processed_df = subset_df.copy()

    return {
        "filename": file.filename,
        "num_rows": len(subset_df),
        "columns": subset_df.columns.tolist(),
        "transformed": subset_df[["ClaimDescription", "DetailedClaimDescription", "Incurred", "InputText", "SeverityLabel"]].to_dict(orient="records")
    }

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ClaimDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

import os
from pathlib import Path

MODEL_DIR = "saved_models/roberta"
LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"

@app.post("/train-model/")
def train_model(force_retrain: bool = False):
    global processed_df

    if processed_df.empty or "InputText" not in processed_df.columns or "SeverityLabel" not in processed_df.columns:
        raise HTTPException(status_code=400, detail="Processed data not found. Upload and process CSV first.")

    # If model already exists and retraining is not forced
    if not force_retrain and os.path.exists(MODEL_DIR) and os.path.exists(LABEL_ENCODER_PATH):
        return {"status": "Model already exists. Use 'force_retrain=true' to retrain."}

    df = processed_df.copy()
    label_encoder = LabelEncoder()
    df["SeverityEncoded"] = label_encoder.fit_transform(df["SeverityLabel"])
    num_labels = len(label_encoder.classes_)

    # Save label encoder
    os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["InputText"].tolist(),
        df["SeverityEncoded"].tolist(),
        test_size=0.2,
        stratify=df["SeverityEncoded"],
        random_state=42
    )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    train_dataset = ClaimDataset(train_encodings, train_labels)
    val_dataset = ClaimDataset(val_encodings, val_labels)

    config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_dir=os.path.join(MODEL_DIR, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics
    )

    train_output = trainer.train()

    # Save model and tokenizer
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    return {
        "status": "training completed and model saved",
        "metrics": train_output.metrics,
        "model_path": MODEL_DIR
    }

from fastapi import UploadFile, File, HTTPException
import os, zipfile, shutil, pickle
from transformers import RobertaForSequenceClassification, RobertaTokenizer

MODEL_UPLOAD_DIR = "uploaded_model"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

def find_model_directory(base_dir: str) -> str:
    """
    Recursively search for a directory containing 'config.json' and either 
    'pytorch_model.bin' or 'model.safetensors'.
    """
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files and ("pytorch_model.bin" in files or "model.safetensors" in files):
            return root
    raise FileNotFoundError("No valid model directory found with config and model weights.")

@app.post("/load-model/")
async def load_model(model_zip: UploadFile = File(...)):
    if not model_zip.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are accepted.")

    # Save uploaded zip temporarily
    zip_path = "temp_model.zip"
    with open(zip_path, "wb") as f:
        f.write(await model_zip.read())

    # Clean old model directory
    if os.path.exists(MODEL_UPLOAD_DIR):
        shutil.rmtree(MODEL_UPLOAD_DIR)
    os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)

    # Extract the zip
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODEL_UPLOAD_DIR)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file.")

    # Try to locate model directory inside uploaded content
    try:
        model_dir = find_model_directory(MODEL_UPLOAD_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Load model and tokenizer
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_dir)
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model/tokenizer: {e}")

    # Load label encoder
    label_encoder_path = os.path.join(model_dir, LABEL_ENCODER_FILENAME)
    if not os.path.exists(label_encoder_path):
        raise HTTPException(status_code=400, detail=f"{LABEL_ENCODER_FILENAME} not found in model directory.")

    try:
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load label encoder: {e}")

    return {
        "status": "Model loaded successfully.",
        "model_dir": model_dir,
        "labels": list(label_encoder.classes_)
    }




from fastapi import UploadFile, File, HTTPException, FastAPI
from pathlib import Path
import os
import zipfile
import shutil
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import stat


POLICY_DOCS_DIR = "policy_doc"
FAISS_INDEX_PATH = "policy_index.faiss"
CHUNK_METADATA_PATH = "chunk_metadata.pkl"

# Helper for deletion
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# ZIP Extract
def extract_zip_to_dir(zip_path, extract_dir):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, onerror=remove_readonly)
    os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# ✅ Sentence splitter without nltk
def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    return sentence_endings.split(text)

# Chunking function
def sentence_chunk_policy_docs(folder_path, chunk_size=3):
    chunks = []
    txt_files = list(Path(folder_path).rglob("*.txt"))
    for file in txt_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = split_into_sentences(text)
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            chunks.append({"content": chunk.strip(), "source": file.name})
    return chunks

# FastAPI endpoint
@app.post("/upload-policy-docs/")
async def upload_policy_docs(zip_file: UploadFile = File(...)):
    if not zip_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")

    # Save the uploaded ZIP
    zip_path = f"{POLICY_DOCS_DIR}.zip"
    with open(zip_path, "wb") as f:
        f.write(await zip_file.read())

    # Extract it
    extract_zip_to_dir(zip_path, POLICY_DOCS_DIR)

    # Sentence chunking
    chunks = sentence_chunk_policy_docs(POLICY_DOCS_DIR)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks found in the uploaded documents.")

    # Embedding
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata
    with open(CHUNK_METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return {
        "message": "Policy documents uploaded, embedded, and indexed successfully.",
        "num_documents": len(list(Path(POLICY_DOCS_DIR).rglob('*.txt'))),
        "num_chunks": len(chunks)
    }