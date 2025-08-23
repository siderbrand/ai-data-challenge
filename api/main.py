from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import joblib, json, numpy as np, pandas as pd
from pathlib import Path

# --- Rutas relativas a la raíz del repo ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS_EXP = ROOT / "models" / "experiments"
MODELS_BASE = ROOT / "models" / "baseline"

# --- Carga artefactos ---
with open(DATA / "classes.txt") as f:
    CLASSES = [l.strip() for l in f if l.strip()]

TFIDF = joblib.load(MODELS_BASE / "tfidf.joblib")

# Cambia este nombre si tu carpeta ganadora es otra
WINNER = "linsvm_C1_C0.5_full5fold"
MODEL_DIR = MODELS_EXP / WINNER

MODEL = joblib.load(MODEL_DIR / "model.joblib")
THR_MAP = json.load(open(MODEL_DIR / "thresholds.json"))
THR = np.array([THR_MAP[c] for c in CLASSES])

def to_proba(model, Xsub):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xsub)
        if isinstance(p, list):
            p = np.vstack([col[:, 1] for col in p]).T
        return p
    scores = model.decision_function(Xsub)
    mn, mx = scores.min(axis=0), scores.max(axis=0)
    mx[mx == mn] = mn[mx == mn] + 1e-9
    return (scores - mn) / (mx - mn)

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="AI Data Challenge | Multilabel API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción, lista de dominios
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    texts: List[str]

class PredictOut(BaseModel):
    labels: List[str]
    proba: Dict[str, float]

@app.get("/health")
def health():
    return {"ok": True, "model": WINNER, "num_classes": len(CLASSES)}

@app.post("/predict", response_model=List[PredictOut])
def predict(inp: PredictIn):
    X = TFIDF.transform(pd.Series(inp.texts).astype(str))
    P = to_proba(MODEL, X)
    Y = (P >= THR).astype(int)

    outs = []
    for i, _ in enumerate(inp.texts):
        labs = [c for j, c in enumerate(CLASSES) if Y[i, j] == 1]
        outs.append({
            "labels": labs,
            "proba": {c: float(P[i, j]) for j, c in enumerate(CLASSES)}
        })
    return outs
