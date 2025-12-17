import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/tweets.csv")
df = df[df["sentiment"] != "neutral"]
df["label"] = df["sentiment"].map({"positive":1,"negative":-1})
df = df[["text","label"]]

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    return t.strip()

df["text"] = df["text"].apply(clean_text)

# ----------------------------
# Train/test split
# ----------------------------
np.random.seed(42)
idx = np.random.permutation(len(df))
split = int(0.8*len(df))
train = df.iloc[idx[:split]]
test  = df.iloc[idx[split:]]

# ----------------------------
# Bag-of-Words
# ----------------------------
vocab = Counter()
for t in train["text"]:
    vocab.update(t.split())

MAX_VOCAB = 3000
vocab = dict(vocab.most_common(MAX_VOCAB))
word_to_idx = {w:i for i,w in enumerate(vocab)}

# ----------------------------
# Save vocab for API
# ----------------------------
joblib.dump(word_to_idx, "word_to_idx_library.pkl")

def vectorize(text):
    v = np.zeros(len(word_to_idx))
    for w in text.split():
        if w in word_to_idx:
            v[word_to_idx[w]] += 1
    return v

X_train = np.array([vectorize(t) for t in train["text"]])
y_train = train["label"].values
X_test  = np.array([vectorize(t) for t in test["text"]])
y_test  = test["label"].values

# ----------------------------
# Train Library SVMs
# ----------------------------
models = {
    "Library Hinge": LinearSVC(loss="hinge", max_iter=3000),
    "Library Squared Hinge": LinearSVC(loss="squared_hinge", max_iter=3000)
}

best_model = None
best_f1 = -1
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    f1 = f1_score(y_test, p)
    acc = accuracy_score(y_test, p)
    print(f"{name} | Acc: {acc:.4f} | F1: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# ----------------------------
# Save best library model
# ----------------------------
joblib.dump(best_model, "best_model_library.pkl")
print(f"Saved best model ({best_name}) with F1={best_f1:.4f} to best_model_library.pkl")
