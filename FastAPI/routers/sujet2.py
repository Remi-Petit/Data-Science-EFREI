from fastapi import APIRouter
from schemas.sujet2 import ChurnData
from ml_models import S2_MODELS
import controllers.sujet2 as controller
import os
import json

router = APIRouter(prefix="/sujet-2", tags=["Sujet 2 – Churn client"])

_STATS_DIR = os.getenv('S2_STATS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'IA', 'Sujet_2', 'models_stats'))
_LABELS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'IA', 'Sujet_2', 'labels.json')


def _load_labels() -> dict:
    if os.path.isfile(_LABELS_FILE):
        with open(_LABELS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


@router.get("/models")
def get_models():
    labels = _load_labels()
    return {"models": [{"name": k, "label": labels.get(k, k)} for k in S2_MODELS]}


@router.get("/stats")
def get_stats():
    if not os.path.isdir(_STATS_DIR):
        return {"stats": {}}
    stats = {}
    for fname in sorted(os.listdir(_STATS_DIR)):
        if fname.endswith('.json'):
            model_key = fname[:-5]
            with open(os.path.join(_STATS_DIR, fname), encoding='utf-8') as f:
                stats[model_key] = json.load(f)
    return {"stats": stats}


@router.post("/predict")
def predict(data: ChurnData):
    return controller.predict(data)
