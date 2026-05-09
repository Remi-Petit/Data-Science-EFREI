import joblib
import json
import os
import sys


def _load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        print(f"[ml_models] Config absente : {config_path}")
        return {}
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def _load_group(models_dir: str, group: dict) -> dict:
    """Charge un groupe {clé: {file, label} | filename}, ignore les fichiers absents."""
    result = {}
    for key, entry in group.items():
        filename = entry["file"] if isinstance(entry, dict) else entry
        path = os.path.join(models_dir, filename)
        if not os.path.isfile(path):
            print(f"[ml_models] Modèle absent, ignoré : {path}")
            continue
        result[key] = joblib.load(path)
    return result


# ── SUJET 1 – Maintenance prédictive ─────────────────────────────────────────

_s1_base = os.getenv(
    'S1_BASE_DIR',
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IA', 'Sujet_1'))
)
if _s1_base not in sys.path:
    sys.path.insert(0, _s1_base)

_s1_models_dir = os.getenv('S1_MODELS_DIR', os.path.join(_s1_base, 'models'))
_s1_config     = _load_config(os.path.join(_s1_base, 'models_config.json'))
S1_MODELS      = _load_group(_s1_models_dir, _s1_config.get('failure_24h', {}))
S1_MODELS_TYPE = _load_group(_s1_models_dir, _s1_config.get('failure_type', {}))
S1_MODELS_RUL  = _load_group(_s1_models_dir, _s1_config.get('rul', {}))

# ── SUJET 2 – Churn client ────────────────────────────────────────────────────

_s2_base = os.getenv(
    'S2_BASE_DIR',
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IA', 'Sujet_2'))
)
_s2_models_dir = os.getenv('S2_MODELS_DIR', os.path.join(_s2_base, 'models'))
_s2_config     = _load_config(os.path.join(_s2_base, 'models_config.json'))
S2_MODELS      = _load_group(_s2_models_dir, _s2_config.get('churn', {}))

# ── SUJET 3 – Marketing ROI ───────────────────────────────────────────────────

_s3_base = os.getenv(
    'S3_BASE_DIR',
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IA', 'Sujet_3'))
)
_s3_models_dir = os.getenv('S3_MODELS_DIR', os.path.join(_s3_base, 'models'))
_s3_config     = _load_config(os.path.join(_s3_base, 'models_config.json'))
S3_REG_MODELS  = _load_group(_s3_models_dir, _s3_config.get('regression', {}))
S3_CLS_MODELS  = _load_group(_s3_models_dir, _s3_config.get('classification', {}))

