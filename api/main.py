from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_TITLE = "AI Data Challenge | Multilabel API"
APP_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]


CLASSES_PATH = Path(os.getenv("CLASSES_PATH", ROOT / "data" / "classes.txt"))
DEFAULT_MODEL_DIR = ROOT / "models" / "experiments" / "linsvm_C1_C0.5_full5fold"

MODEL_DIR = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
MODEL_PATH = MODEL_DIR / "model.joblib"
THR_PATH = MODEL_DIR / "thresholds.json"
VEC_PATH = Path(os.getenv("VEC_PATH", MODEL_DIR / "tfidf.joblib"))

def _read_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró classes.txt en {path}")
    with open(path, "r", encoding="utf-8") as f:
        out = [ln.strip() for ln in f if ln.strip()]
    if not out:
        raise ValueError("classes.txt está vacío")
    return out


def _read_thresholds(path: Path, classes: List[str]) -> np.ndarray:
    if not path.exists():
        # Si no hay thresholds.json, usar 0.5
        return np.full(len(classes), 0.5, dtype="float64")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.array([float(d.get(c, 0.5)) for c in classes], dtype="float64")


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
if not VEC_PATH.exists():
    raise FileNotFoundError(
        f"Vectorizador TF-IDF no encontrado: {VEC_PATH}\n"
        f"Coloca tfidf.joblib junto a model.joblib en {MODEL_DIR}\n"
        f"o define VEC_PATH=... en variables de entorno."
    )


CLASSES: List[str] = _read_classes(CLASSES_PATH)
VECT = joblib.load(VEC_PATH)
MODEL = joblib.load(MODEL_PATH)
THR = _read_thresholds(THR_PATH, CLASSES)

_re_nonword = re.compile(r"[^a-z0-9\s]+")


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = _re_nonword.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_proba(model, Xsub) -> np.ndarray:
    """
    Devuelve probabilidades (n_samples, n_classes).
    - Si el modelo tiene predict_proba -> usarlo (OneVsRest puede devolver lista).
    - Si no, aplicar sigmoide sobre decision_function (p.ej., LinearSVC).
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xsub)
        if isinstance(p, list):  # OvR con lista de estimadores
            p = np.column_stack([col[:, 1] for col in p])
        return p.astype("float64")

    scores = model.decision_function(Xsub).astype("float64")
    return 1.0 / (1.0 + np.exp(-scores))


def infer(texts: List[str]) -> List[Dict[str, Any]]:
    if not texts:
        return []
    texts_norm = [normalize_text(t) for t in texts]
    X = VECT.transform(texts_norm)
    proba = to_proba(MODEL, X)

    results: List[Dict[str, Any]] = []
    for i in range(proba.shape[0]):
        row = proba[i, :]
        chosen = [CLASSES[j] for j, p in enumerate(row) if p >= THR[j]]
        if not chosen:
            chosen = [CLASSES[int(np.argmax(row))]]
        results.append(
            {
                "labels": chosen,
                "proba": {c: float(row[j]) for j, c in enumerate(CLASSES)},
            }
        )
    return results


app = FastAPI(title=APP_TITLE, version=APP_VERSION)

origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
allow_origins = (
    ["*"] if origins_env == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]
)

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
    proba: Dict[str, float]


@app.get("/")
def root():
    return {"ok": True, "message": "API OK. Visita /docs para probar."}


@app.get("/health")
def health():
    return {
        "ok": True,
        "app": APP_TITLE,
        "version": APP_VERSION,
        "n_classes": len(CLASSES),
        "classes": CLASSES,
        "has_predict_proba": bool(hasattr(MODEL, "predict_proba")),
        "model_dir": str(MODEL_DIR),
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VEC_PATH),
        "thresholds_path": str(THR_PATH),
    }


@app.post("/predict", response_model=List[PredictOut])
def predict(payload: PredictIn):
    if not isinstance(payload.texts, list):
        raise HTTPException(status_code=400, detail="`texts` debe ser una lista de strings")
    try:
        return infer(payload.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {e}")


@app.get("/_diag")
def diag():
    return {
        "classes": CLASSES,
        "thr": [float(x) for x in THR],
        "model_type": str(type(MODEL)),
        "model_dir": str(MODEL_DIR),
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VEC_PATH),
        "has_predict_proba": hasattr(MODEL, "predict_proba"),
        "allowed_origins": allow_origins,
    }


