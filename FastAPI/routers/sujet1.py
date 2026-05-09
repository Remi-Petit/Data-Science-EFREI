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

_RANK_COLS = {"failure_24h": "F1-score", "failure_type": "F1-score", "rul": "R²"}


def _load_stats_group(task: str) -> dict:
    task_dir = os.path.join(_STATS_DIR, task)
    result = {}
    if not os.path.isdir(task_dir):
        return result
    for fname in os.listdir(task_dir):
        if fname.endswith('.json'):
            with open(os.path.join(task_dir, fname), encoding='utf-8') as f:
                result[fname[:-5]] = json.load(f)
    return result


@router.get("/models")
def get_models():
    """Retourne tous les modèles groupés par tâche, triés du meilleur au moins bon."""
    if not os.path.isfile(_CONFIG_FILE):
        return {"models": {}}
    with open(_CONFIG_FILE, encoding='utf-8') as f:
        config = json.load(f)
    result = {}
    for group_name, entries in config.items():
        rank_col = _RANK_COLS.get(group_name, "F1-score")
        grp_stats = _load_stats_group(group_name)
        models = [
            {"name": k, "label": v["label"] if isinstance(v, dict) and "label" in v else k}
            for k, v in entries.items()
        ]
        models.sort(key=lambda m: grp_stats.get(m["name"], {}).get(rank_col, 0), reverse=True)
        result[group_name] = models
    return {"models": result}


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
