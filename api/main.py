from __future__ import annotations
import os, re, json, time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_TITLE = "AI Data Challenge | Multilabel API"
APP_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]

CLASSES_PATH = Path(os.getenv("CLASSES_PATH", ROOT / "data" / "classes.txt"))


DEFAULT_MODEL_DIR = ROOT / "models" / "experiments" / "ensemble_blend"
MODEL_DIR = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
MODEL_PATH = MODEL_DIR / "model.joblib"
VEC_PATH   = Path(os.getenv("VEC_PATH", MODEL_DIR / "tfidf.joblib"))
THR_PATH   = MODEL_DIR / "thresholds.json"


SHORT_TO_LONG = {
    "cardio": "cardiovascular",
    "hepato": "hepatorenal",
    "neuro":  "neurological",
    "onco":   "oncological",
}
LONG_TO_SHORT = {v: k for k, v in SHORT_TO_LONG.items()}
SHORT_ORDER = ["cardio", "hepato", "neuro", "onco"]  # orden en la UI

def _read_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró classes.txt en {path}")
    with open(path, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    if not classes:
        raise ValueError("classes.txt está vacío")
    return classes

def _read_thresholds(path: Path, classes: List[str]) -> np.ndarray:
    if not path.exists():
        # por si falta el archivo en prod: umbral seguro
        return np.full(len(classes), 0.5, dtype="float64")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.array([float(d.get(c, 0.5)) for c in classes], dtype="float64")

CLASSES: List[str] = _read_classes(CLASSES_PATH)
MODEL = joblib.load(MODEL_PATH)
VECT  = joblib.load(VEC_PATH)
THR   = _read_thresholds(THR_PATH, CLASSES)

# normalización simple (alineada con el TF-IDF de entrenamiento)
_re_nonword = re.compile(r"[^a-z0-9\s]+")
def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = _re_nonword.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_proba(model, Xsub) -> np.ndarray:
    
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xsub)
        if isinstance(p, list):  # OvR devuelve lista por clase
            p = np.column_stack([col[:, 1] for col in p])
        return p.astype("float64")
    # fallback: sigmoide de decision_function (LinearSVC, etc.)
    scores = model.decision_function(Xsub).astype("float64")
    return 1.0 / (1.0 + np.exp(-scores))

def infer(texts: List[str]) -> List[Dict[str, Any]]:
    if not texts:
        return []

    texts_norm = [normalize_text(t) for t in texts]
    X = VECT.transform(texts_norm)
    proba = to_proba(MODEL, X)  # (n, C)

    results = []
    for i in range(proba.shape[0]):
        row = proba[i, :]
        # labels con nombres largos (para trazabilidad)
        chosen_long = [CLASSES[j] for j, p in enumerate(row) if p >= THR[j]]
        if not chosen_long:
            chosen_long = [CLASSES[int(np.argmax(row))]]

        # proba en ambos formatos
        proba_long  = {CLASSES[j]: float(row[j]) for j in range(len(CLASSES))}
        proba_short = {LONG_TO_SHORT[CLASSES[j]]: float(row[j]) for j in range(len(CLASSES))}
        labels_short = [LONG_TO_SHORT[c] for c in chosen_long]

        results.append({
            "labels": chosen_long,        # largos
            "labels_short": labels_short, # cortos p/ UI
            "proba": proba_short,         # <── la UI de V0 debe leer ESTO
            "proba_full": proba_long,     # y esto queda para debug/registro
        })
    return results

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
allow_origins = ["*"] if origins_env == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False if allow_origins == ["*"] else True,
)

class PredictIn(BaseModel):
    texts: List[str] = Field(..., description="Lista de abstracts (uno por elemento)")

class PredictOut(BaseModel):
    labels: List[str]
    labels_short: List[str]
    proba: Dict[str, float]
    proba_full: Dict[str, float]

@app.get("/health")
def health():
    return {
        "ok": True,
        "app": APP_TITLE,
        "version": APP_VERSION,
        "n_classes": len(CLASSES),
        "classes_long": CLASSES,
        "classes_short": [LONG_TO_SHORT[c] for c in CLASSES],
        "has_predict_proba": bool(hasattr(MODEL, "predict_proba")),
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VEC_PATH),
    }

@app.post("/predict", response_model=List[PredictOut])
def predict(payload: PredictIn):
    t0 = time.time()
    if not isinstance(payload.texts, list):
        raise HTTPException(status_code=400, detail="`texts` debe ser una lista de strings")
    try:
        res = infer(payload.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {e}")
    _ = time.time() - t0
    return res

@app.get("/_diag")
def diag():
    return {
        "classes": CLASSES,
        "classes_short": [LONG_TO_SHORT[c] for c in CLASSES],
        "thr": [float(x) for x in THR],
        "model_type": str(type(MODEL)),
        "model_dir": str(MODEL_DIR),
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VEC_PATH),
        "has_predict_proba": hasattr(MODEL, "predict_proba"),
        "allowed_origins": allow_origins,
    }
