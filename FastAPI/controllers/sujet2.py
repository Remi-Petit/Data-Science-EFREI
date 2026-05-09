import pandas as pd
from schemas.sujet2 import ChurnData, S2_CAT_MAPS
from ml_models import S2_MODELS


def _preprocess(data: ChurnData) -> pd.DataFrame:
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
        "gender_enc":                 S2_CAT_MAPS["gender"][data.gender],
        "customer_segment_enc":       S2_CAT_MAPS["customer_segment"][data.customer_segment],
        "signup_channel_enc":         S2_CAT_MAPS["signup_channel"][data.signup_channel],
        "contract_type_enc":          S2_CAT_MAPS["contract_type"][data.contract_type],
        "payment_method_enc":         S2_CAT_MAPS["payment_method"][data.payment_method],
        "discount_applied_enc":       S2_CAT_MAPS["discount_applied"][data.discount_applied],
        "price_increase_last_3m_enc": S2_CAT_MAPS["price_increase_last_3m"][data.price_increase_last_3m],
        "survey_response_enc":        S2_CAT_MAPS["survey_response"][data.survey_response],
        "complaint_type_enc":         S2_CAT_MAPS["complaint_type"][data.complaint_type],
        # Features engineered
        "revenue_per_month": data.total_revenue / (data.tenure_months + 1),
        "engagement_score":  data.weekly_active_days * data.avg_session_time,
        "ticket_burden":     data.support_tickets * (data.avg_resolution_time + 1),
    }
    return pd.DataFrame([row])


def predict(data: ChurnData) -> dict:
    df = _preprocess(data)

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
