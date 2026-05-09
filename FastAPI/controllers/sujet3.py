import pandas as pd
from schemas.sujet3 import MarketingData, S3_INFLUENCER_MAP, S3_PERF_LOW, S3_PERF_HIGH
from ml_models import S3_REG_MODELS


def _preprocess(data: MarketingData) -> pd.DataFrame:
    total_budget = data.tv + data.radio + data.social_media
    row = {
        "tv":                    data.tv,
        "radio":                 data.radio,
        "social_media":          data.social_media,
        "influencer_enc":        S3_INFLUENCER_MAP[data.influencer],
        "total_budget":          total_budget,
        "tv_share":              data.tv           / total_budget if total_budget > 0 else 0,
        "radio_share":           data.radio        / total_budget if total_budget > 0 else 0,
        "social_share":          data.social_media / total_budget if total_budget > 0 else 0,
        "tv_social_interaction": data.tv * data.social_media,
    }
    return pd.DataFrame([row])


def predict(data: MarketingData) -> dict:
    df = _preprocess(data)

    results = {}
    for model_name in data.models:
        model = S3_REG_MODELS[model_name]
        sales_pred = float(model.predict(df)[0])

        if sales_pred < S3_PERF_LOW:
            perf_label = "Low"
        elif sales_pred < S3_PERF_HIGH:
            perf_label = "Medium"
        else:
            perf_label = "High"

        total_budget = data.tv + data.radio + data.social_media
        roi = round(sales_pred / total_budget, 4) if total_budget > 0 else None

        results[model_name] = {
            "sales_prediction": round(sales_pred, 4),
            "performance":      perf_label,
            "roi_estimate":     roi,
        }

    return {"results": results}
