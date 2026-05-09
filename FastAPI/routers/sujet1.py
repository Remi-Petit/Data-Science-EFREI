from fastapi import APIRouter
from schemas.sujet1 import MachineData
from ml_models import S1_MODELS
import controllers.sujet1 as controller
import os
import json

router = APIRouter(prefix="/sujet-1", tags=["Sujet 1 – Maintenance prédictive"])

_S1_BASE    = os.getenv('S1_BASE_DIR', os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'IA', 'Sujet_1')))
_STATS_DIR  = os.getenv('S1_STATS_DIR', os.path.join(_S1_BASE, 'models_stats'))
_CONFIG_FILE = os.path.join(_S1_BASE, 'models_config.json')


def _load_labels() -> dict:
    if not os.path.isfile(_CONFIG_FILE):
        return {}
    with open(_CONFIG_FILE, encoding='utf-8') as f:
        config = json.load(f)
    labels = {}
    for group in config.values():
        for key, entry in group.items():
            if isinstance(entry, dict) and "label" in entry:
                labels[key] = entry["label"]
    return labels


@router.get("/models")
def get_models():
    labels = _load_labels()
    return {"models": [{"name": k, "label": labels.get(k, k)} for k in S1_MODELS]}


@router.get("/stats")
def get_stats():
    result = {"failure_24h": {}, "failure_type": {}, "rul": {}}
    for task in ("failure_24h", "failure_type", "rul"):
        task_dir = os.path.join(_STATS_DIR, task)
        if not os.path.isdir(task_dir):
            continue
        for fname in sorted(os.listdir(task_dir)):
            if fname.endswith('.json'):
                model_key = fname[:-5]
                with open(os.path.join(task_dir, fname), encoding='utf-8') as f:
                    result[task][model_key] = json.load(f)
    return {"stats": result}


@router.post("/predict")
def predict(data: MachineData):
    return controller.predict(data)
