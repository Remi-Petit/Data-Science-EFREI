from pydantic import BaseModel, field_validator
from typing import List, Literal

S2_CAT_MAPS = {
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
    models: List[str] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_valid(cls, v):
        from ml_models import S2_MODELS
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        v = list(dict.fromkeys(v))
        unavailable = [m for m in v if m not in S2_MODELS]
        if unavailable:
            raise ValueError(f"Modèle(s) non disponible(s) : {unavailable}. Disponibles : {list(S2_MODELS.keys())}")
        return v
