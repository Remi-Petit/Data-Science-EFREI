import joblib
import os
import sys

# ── SUJET 1 – Maintenance prédictive ─────────────────────────────────────────

_s1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IA', 'Sujet_1')
if _s1_dir not in sys.path:
    sys.path.insert(0, _s1_dir)

_s1_models_dir = os.getenv('S1_MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'IA', 'Sujet_1', 'models'))
S1_MODELS = {
    "logistic_regression": joblib.load(os.path.join(_s1_models_dir, 'logistic_regression_failure_24h.joblib')),
    "random_forest":       joblib.load(os.path.join(_s1_models_dir, 'random_forest_failure_24h.joblib')),
    "xgboost":             joblib.load(os.path.join(_s1_models_dir, 'xgboost_failure_24h.joblib')),
}
S1_MODELS_TYPE = {
    "logistic_regression": joblib.load(os.path.join(_s1_models_dir, 'logistic_regression_failure_type.joblib')),
    "random_forest":       joblib.load(os.path.join(_s1_models_dir, 'random_forest_failure_type.joblib')),
    "xgboost":             joblib.load(os.path.join(_s1_models_dir, 'xgboost_failure_type.joblib')),
}

# ── SUJET 2 – Churn client ────────────────────────────────────────────────────

_s2_models_dir = os.getenv('S2_MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'IA', 'Sujet_2', 'models'))
S2_MODELS = {
    "logistic_regression": joblib.load(os.path.join(_s2_models_dir, 'logistic_regression_churn.joblib')),
    "random_forest":       joblib.load(os.path.join(_s2_models_dir, 'random_forest_churn.joblib')),
    "xgboost":             joblib.load(os.path.join(_s2_models_dir, 'xgboost_churn.joblib')),
    "mlp":                 joblib.load(os.path.join(_s2_models_dir, 'mlp_churn.joblib')),
}

# ── SUJET 3 – Marketing ROI ───────────────────────────────────────────────────

_s3_models_dir = os.getenv('S3_MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'IA', 'Sujet_3', 'models'))
S3_REG_MODELS = {
    "linear_regression": joblib.load(os.path.join(_s3_models_dir, 'linear_regression_sales.joblib')),
    "random_forest":     joblib.load(os.path.join(_s3_models_dir, 'random_forest_sales.joblib')),
    "xgboost":           joblib.load(os.path.join(_s3_models_dir, 'xgboost_sales.joblib')),
    "mlp":               joblib.load(os.path.join(_s3_models_dir, 'mlp_sales.joblib')),
}
S3_CLS_MODELS = {
    "random_forest": joblib.load(os.path.join(_s3_models_dir, 'random_forest_performance.joblib')),
    "xgboost":       joblib.load(os.path.join(_s3_models_dir, 'xgboost_performance.joblib')),
}
