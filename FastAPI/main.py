from fastapi import FastAPI
import joblib
import os
import sys
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List, Literal

app = FastAPI()

# ── SUJET 1 – Maintenance prédictive ─────────────────────────────────────────

# Rendre le module src (IA/Sujet_1/src) importable par pickle lors du chargement des modèles
_s1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IA', 'Sujet_1')
if _s1_dir not in sys.path:
    sys.path.insert(0, _s1_dir)

S1ModelName = Literal["logistic_regression", "random_forest", "xgboost"]

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

class MachineData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    ambient_temp: float
    machine_type_enc: int
    operating_mode_enc: int
    hour: int
    dayofweek: int
    month: int
    models: List[S1ModelName] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))


# ── SUJET 2 – Churn client ────────────────────────────────────────────────────

S2ModelName = Literal["logistic_regression", "random_forest", "xgboost", "mlp"]

_s2_models_dir = os.getenv('S2_MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'IA', 'Sujet_2', 'models'))
S2_MODELS = {
    "logistic_regression": joblib.load(os.path.join(_s2_models_dir, 'logistic_regression_churn.joblib')),
    "random_forest":       joblib.load(os.path.join(_s2_models_dir, 'random_forest_churn.joblib')),
    "xgboost":             joblib.load(os.path.join(_s2_models_dir, 'xgboost_churn.joblib')),
    "mlp":                 joblib.load(os.path.join(_s2_models_dir, 'mlp_churn.joblib')),
}

# Encodage label (ordre alphabétique, miroir du LabelEncoder sklearn)
_S2_CAT_MAPS = {
    "gender":                 {"Female": 0, "Male": 1},
    "customer_segment":       {"Enterprise": 0, "Individual": 1, "SME": 2},
    "signup_channel":         {"Mobile": 0, "Referral": 1, "Web": 2},
    "contract_type":          {"Monthly": 0, "Quarterly": 1, "Yearly": 2},
    "payment_method":         {"Bank Transfer": 0, "Card": 1, "PayPal": 2},
    "discount_applied":       {"No": 0, "Yes": 1},
    "price_increase_last_3m": {"No": 0, "Yes": 1},
    "survey_response":        {"Neutral": 0, "Satisfied": 1, "Unsatisfied": 2},
    "complaint_type":         {"Billing": 0, "Service": 1, "Technical": 2, "Unknown": 3},
}

class ChurnData(BaseModel):
    # Variables numériques
    age: float
    tenure_months: float
    monthly_logins: float
    weekly_active_days: float
    avg_session_time: float
    features_used: float
    usage_growth_rate: float
    last_login_days_ago: float
    monthly_fee: float
    total_revenue: float
    payment_failures: float
    support_tickets: float
    avg_resolution_time: float
    csat_score: float
    escalations: float
    email_open_rate: float
    marketing_click_rate: float
    nps_score: float
    referral_count: float
    # Variables catégorielles
    gender: Literal["Female", "Male"]
    customer_segment: Literal["Enterprise", "Individual", "SME"]
    signup_channel: Literal["Mobile", "Referral", "Web"]
    contract_type: Literal["Monthly", "Quarterly", "Yearly"]
    payment_method: Literal["Bank Transfer", "Card", "PayPal"]
    discount_applied: Literal["No", "Yes"]
    price_increase_last_3m: Literal["No", "Yes"]
    survey_response: Literal["Neutral", "Satisfied", "Unsatisfied"]
    complaint_type: Literal["Billing", "Service", "Technical", "Unknown"] = "Unknown"
    models: List[S2ModelName] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))


def _preprocess_churn(data: ChurnData) -> pd.DataFrame:
    row = {
        "age": data.age,
        "tenure_months": data.tenure_months,
        "monthly_logins": data.monthly_logins,
        "weekly_active_days": data.weekly_active_days,
        "avg_session_time": data.avg_session_time,
        "features_used": data.features_used,
        "usage_growth_rate": data.usage_growth_rate,
        "last_login_days_ago": data.last_login_days_ago,
        "monthly_fee": data.monthly_fee,
        "total_revenue": data.total_revenue,
        "payment_failures": data.payment_failures,
        "avg_resolution_time": data.avg_resolution_time,
        "csat_score": data.csat_score,
        "escalations": data.escalations,
        "email_open_rate": data.email_open_rate,
        "marketing_click_rate": data.marketing_click_rate,
        "nps_score": data.nps_score,
        "referral_count": data.referral_count,
        # Catégorielles encodées
        "gender_enc":                 _S2_CAT_MAPS["gender"][data.gender],
        "customer_segment_enc":       _S2_CAT_MAPS["customer_segment"][data.customer_segment],
        "signup_channel_enc":         _S2_CAT_MAPS["signup_channel"][data.signup_channel],
        "contract_type_enc":          _S2_CAT_MAPS["contract_type"][data.contract_type],
        "payment_method_enc":         _S2_CAT_MAPS["payment_method"][data.payment_method],
        "discount_applied_enc":       _S2_CAT_MAPS["discount_applied"][data.discount_applied],
        "price_increase_last_3m_enc": _S2_CAT_MAPS["price_increase_last_3m"][data.price_increase_last_3m],
        "survey_response_enc":        _S2_CAT_MAPS["survey_response"][data.survey_response],
        "complaint_type_enc":         _S2_CAT_MAPS["complaint_type"][data.complaint_type],
        # Features engineered
        "revenue_per_month": data.total_revenue / (data.tenure_months + 1),
        "engagement_score":  data.weekly_active_days * data.avg_session_time,
        "ticket_burden":     data.support_tickets * (data.avg_resolution_time + 1),
    }
    return pd.DataFrame([row])


# ── ENDPOINTS SUJET 1 & 2 ─────────────────────────────────────────────────────

@app.post("/sujet-1/predict")
def predict_s1(data: MachineData):
    features = data.model_dump(exclude={"models"})
    df = pd.DataFrame([features])

    results = {}
    for model_name in data.models:
        model = S1_MODELS[model_name]
        prediction = int(model.predict(df)[0])
        probabilite = float(model.predict_proba(df)[0][1])
        result = {
            "prediction": prediction,
            "label": "Panne probable" if prediction == 1 else "Pas de panne",
            "probabilite_panne": round(probabilite, 4),
        }
        if prediction == 1:
            model_type = S1_MODELS_TYPE[model_name]
            type_proba = model_type.predict_proba(df)[0]
            type_classes = model_type.classes_
            all_scores = {cls: round(float(p), 4) for cls, p in zip(type_classes, type_proba)}
            failure_scores = {cls: p for cls, p in all_scores.items() if cls != 'none'}
            result["cause_potentielle"] = max(failure_scores, key=failure_scores.get)
            result["probabilites_causes"] = all_scores
        results[model_name] = result

    return {"results": results}

@app.post("/sujet-2/predict")
def predict_s2(data: ChurnData):
    df = _preprocess_churn(data)

    results = {}
    for model_name in data.models:
        model = S2_MODELS[model_name]
        prediction = int(model.predict(df)[0])
        probabilite = float(model.predict_proba(df)[0][1])
        results[model_name] = {
            "prediction": prediction,
            "label": "Churn probable" if prediction == 1 else "Client fidèle",
            "probabilite_churn": round(probabilite, 4),
        }

    return {"results": results}

# ── SUJET 3 – Marketing ROI ───────────────────────────────────────────────────

S3RegModelName = Literal["linear_regression", "random_forest", "xgboost", "mlp"]
S3ClsModelName = Literal["random_forest", "xgboost"]

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

# Encodage Influencer (ordre alphabétique, miroir LabelEncoder sklearn)
_S3_INFLUENCER_MAP = {'Macro': 0, 'Mega': 1, 'Micro': 2, 'Nano': 3}
_S3_PERF_LABELS    = {0: 'Low', 1: 'Medium', 2: 'High'}

# Seuils de classification (quantiles 33 % / 66 % du dataset)
_S3_PERF_LOW  = 136.86
_S3_PERF_HIGH = 241.53

class MarketingData(BaseModel):
    tv:           float
    radio:        float
    social_media: float
    influencer:   Literal["Macro", "Mega", "Micro", "Nano"]
    models: List[S3RegModelName] = ["linear_regression"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))


def _preprocess_marketing(data: MarketingData) -> pd.DataFrame:
    total_budget = data.tv + data.radio + data.social_media
    row = {
        "tv":                   data.tv,
        "radio":                data.radio,
        "social_media":         data.social_media,
        "influencer_enc":       _S3_INFLUENCER_MAP[data.influencer],
        "total_budget":         total_budget,
        "tv_share":             data.tv           / total_budget if total_budget > 0 else 0,
        "radio_share":          data.radio        / total_budget if total_budget > 0 else 0,
        "social_share":         data.social_media / total_budget if total_budget > 0 else 0,
        "tv_social_interaction": data.tv * data.social_media,
    }
    return pd.DataFrame([row])


@app.post("/sujet-3/predict")
def predict_s3(data: MarketingData):
    df = _preprocess_marketing(data)

    results = {}
    for model_name in data.models:
        model = S3_REG_MODELS[model_name]
        sales_pred = float(model.predict(df)[0])

        # Classification automatique de la performance
        if sales_pred < _S3_PERF_LOW:
            perf_label = "Low"
        elif sales_pred < _S3_PERF_HIGH:
            perf_label = "Medium"
        else:
            perf_label = "High"

        # ROI estimé = ventes / budget total
        total_budget = data.tv + data.radio + data.social_media
        roi = round(sales_pred / total_budget, 4) if total_budget > 0 else None

        results[model_name] = {
            "sales_prediction": round(sales_pred, 4),
            "performance":      perf_label,
            "roi_estimate":     roi,
        }

    return {"results": results}


# ── ENDPOINTS GLOBAUX ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "sujet_1_models": list(S1_MODELS.keys()),
        "sujet_2_models": list(S2_MODELS.keys()),
        "sujet_3_reg_models": list(S3_REG_MODELS.keys()),
        "sujet_3_cls_models": list(S3_CLS_MODELS.keys()),
    }


# Rétrocompatibilité
@app.post("/predict")
def predict_legacy(data: MachineData):
    return predict_s1(data)