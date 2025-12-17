import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/tweets.csv")

df = df[df["sentiment"] != "neutral"]
df["label"] = df["sentiment"].map({"positive": 1, "negative": -1})
df = df[["text", "label"]]

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
split = int(0.8 * len(df))

train = df.iloc[idx[:split]]
test  = df.iloc[idx[split:]]

# ----------------------------
# Bag of Words
# ----------------------------
vocab = Counter()
for t in train["text"]:
    vocab.update(t.split())

MAX_VOCAB = 3000
vocab = dict(vocab.most_common(MAX_VOCAB))
word_to_idx = {w:i for i,w in enumerate(vocab)}

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
# SVM functions
# ----------------------------
def score(x,w,b): return np.dot(w,x)+b

def hinge(y,s): return max(0,1-y*s)
def sq_hinge(y,s): return max(0,1-y*s)**2
def log_loss(y,s): return np.log(1+np.exp(-y*s))

def hinge_grad(x,y,s):
    return (np.zeros_like(x),0) if y*s>=1 else (-y*x,-y)

def sq_grad(x,y,s):
    m=1-y*s
    return (np.zeros_like(x),0) if m<=0 else (-2*y*m*x,-2*y*m)

def log_grad(x,y,s):
    p=1/(1+np.exp(y*s))
    return -y*p*x,-y*p

# ----------------------------
# Training
# ----------------------------
def train(loss, lr=0.001, epochs=25):
    w=np.zeros(X_train.shape[1])
    b=0
    L=[]
    for e in range(epochs):
        tot=0
        for x,y in zip(X_train,y_train):
            s=score(x,w,b)
            if loss=="hinge":
                l=hinge(y,s); dw,db=hinge_grad(x,y,s)
            elif loss=="squared":
                l=sq_hinge(y,s); dw,db=sq_grad(x,y,s)
            else:
                l=log_loss(y,s); dw,db=log_grad(x,y,s)
            w-=lr*dw; b-=lr*db; tot+=l
        L.append(tot/len(X_train))
        print(loss,"epoch",e+1,"loss",L[-1])
    return w,b,L

w_h,b_h,l_h = train("hinge")
w_s,b_s,l_s = train("squared")
w_l,b_l,l_l = train("logistic")

# ----------------------------
# Evaluation
# ----------------------------
def predict(X,w,b): return np.sign(np.dot(X,w)+b)

def metrics(y,p):
    tp=np.sum((y==1)&(p==1))
    tn=np.sum((y==-1)&(p==-1))
    fp=np.sum((y==-1)&(p==1))
    fn=np.sum((y==1)&(p==-1))
    acc=(tp+tn)/len(y)
    prec=tp/(tp+fp+1e-8)
    rec=tp/(tp+fn+1e-8)
    f1=2*prec*rec/(prec+rec+1e-8)
    return acc,prec,rec,f1

models = {
    "Hinge":(w_h,b_h),
    "Squared":(w_s,b_s),
    "Logistic":(w_l,b_l)
}

for name,(w,b) in models.items():
    p=predict(X_test,w,b)
    a,pr,re,f=metrics(y_test,p)
    print(name,"| Acc:",a,"| F1:",f)

# ----------------------------
# Plot
# ----------------------------
plt.plot(l_h,label="Hinge")
plt.plot(l_s,label="Squared")
plt.plot(l_l,label="Logistic")
plt.legend()
plt.title("Manual SVM Training Loss")
plt.savefig("manual_training_curve.png")
plt.show()
