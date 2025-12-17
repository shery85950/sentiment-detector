from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re

# ----------------------------
# Input schema
# ----------------------------
class TextInput(BaseModel):
    text: str

# ----------------------------
# Load best model
# ----------------------------
# Change the filename to either manual or library best model
best_model_file = "best_model_manual.pkl"  # or "best_model.pkl" for library
w, b = joblib.load(best_model_file)

# ----------------------------
# Load vocab used for vectorization
# ----------------------------
# This should match the vocab from training
# Here we rebuild it from a saved file or recreate from your script
# For demo, define a small example vocab. Replace with your actual vocab.
word_to_idx = joblib.load("word_to_idx.pkl")  # save this during training

# ----------------------------
# Text preprocessing
# ----------------------------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    return t.strip()

def vectorize(text):
    v = np.zeros(len(word_to_idx))
    for w in text.split():
        if w in word_to_idx:
            v[word_to_idx[w]] += 1
    return v

def predict(text):
    X = np.array([vectorize(clean_text(text))])
    s = np.dot(X, w) + b
    pred = np.sign(s)[0]
    return int(pred)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="SVM Sentiment API")

@app.post("/predict")
def predict_endpoint(input: TextInput):
    label = predict(input.text)
    sentiment = "positive" if label==1 else "negative"
    return {"prediction": label, "sentiment": sentiment}
