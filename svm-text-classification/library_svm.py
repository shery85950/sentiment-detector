import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/tweets.csv")
df = df[df["sentiment"] != "neutral"]
df["label"] = df["sentiment"].map({"positive":1,"negative":-1})
df = df[["text","label"]]

def clean_text(t):
    t=t.lower()
    t=re.sub(r"http\S+","",t)
    t=re.sub(r"[^a-z\s]","",t)
    return t.strip()

df["text"] = df["text"].apply(clean_text)

# ----------------------------
# Split
# ----------------------------
np.random.seed(42)
idx=np.random.permutation(len(df))
split=int(0.8*len(df))
train=df.iloc[idx[:split]]
test=df.iloc[idx[split:]]

# ----------------------------
# BoW
# ----------------------------
vocab=Counter()
for t in train["text"]:
    vocab.update(t.split())

vocab=dict(vocab.most_common(3000))
word_to_idx={w:i for i,w in enumerate(vocab)}

def vectorize(text):
    v=np.zeros(len(word_to_idx))
    for w in text.split():
        if w in word_to_idx:
            v[word_to_idx[w]]+=1
    return v

X_train=np.array([vectorize(t) for t in train["text"]])
y_train=train["label"].values
X_test=np.array([vectorize(t) for t in test["text"]])
y_test=test["label"].values

# ----------------------------
# Library SVMs
# ----------------------------
models = {
    "Library Hinge": LinearSVC(loss="hinge", max_iter=3000),
    "Library Squared Hinge": LinearSVC(loss="squared_hinge", max_iter=3000)
}

for name,model in models.items():
    model.fit(X_train,y_train)
    p=model.predict(X_test)
    print(
        name,
        "Acc:",accuracy_score(y_test,p),
        "F1:",f1_score(y_test,p)
    )
