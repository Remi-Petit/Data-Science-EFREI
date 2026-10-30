use crate::{error::*, models::*};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::{path::PathBuf, sync::Arc};

pub struct Sujet2 {
    stats_dir: PathBuf,
    config: Map<String, Value>,
    pub churn: ModelGroup,
}

impl Sujet2 {
    pub fn load() -> Self {
        let base = env_path("S2_BASE_DIR", || PathBuf::from("IA/Sujet_2"));
        let models_dir = env_path("S2_MODELS_DIR", || base.join("models"));
        let stats_dir = env_path("S2_STATS_DIR", || base.join("models_stats"));
        let config = load_config(&base.join("models_config.json"));
        Self {
            churn: load_group(&models_dir, config.get("churn")),
            stats_dir,
            config,
        }
    }
}

pub fn router() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/models", get(models))
        .route("/stats", get(stats))
        .route("/predict", post(predict))
}

async fn models(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    let s = &st.s2;
    let labels = labels_of(&s.config);
    let list: Vec<Value> = s
        .churn
        .keys()
        .into_iter()
        .map(|k| json!({ "name": k, "label": labels.get(&k).cloned().unwrap_or_else(|| json!(k)) }))
        .collect();
    Ok(Json(json!({ "models": list })))
}

async fn stats(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    Ok(Json(json!({ "stats": stats_map(&st.s2.stats_dir) })))
}

fn default_models() -> Vec<String> {
    vec!["random_forest".into()]
}

fn default_complaint() -> String {
    "Unknown".into()
}

#[derive(Deserialize)]
struct ChurnData {
    age: f64,
    tenure_months: f64,
    monthly_logins: f64,
    weekly_active_days: f64,
    avg_session_time: f64,
    features_used: f64,
    usage_growth_rate: f64,
    last_login_days_ago: f64,
    monthly_fee: f64,
    total_revenue: f64,
    payment_failures: f64,
    support_tickets: f64,
    avg_resolution_time: f64,
    csat_score: f64,
    escalations: f64,
    email_open_rate: f64,
    marketing_click_rate: f64,
    nps_score: f64,
    referral_count: f64,
    gender: String,
    customer_segment: String,
    signup_channel: String,
    contract_type: String,
    payment_method: String,
    discount_applied: String,
    price_increase_last_3m: String,
    survey_response: String,
    #[serde(default = "default_complaint")]
    complaint_type: String,
    #[serde(default = "default_models")]
    models: Vec<String>,
}

fn enc(field: &str, value: &str, map: &[(&str, f32)]) -> Result<f32, ApiError> {
    map.iter()
        .find(|(k, _)| *k == value)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let allowed: Vec<_> = map.iter().map(|(k, _)| *k).collect();
            ApiError::unprocessable(format!("{field} invalide : {value}. Valeurs : {allowed:?}"))
        })
}

async fn predict(State(st): State<Arc<crate::AppState>>, Json(d): Json<ChurnData>) -> ApiResult {
    let s = &st.s2;
    let names = validate_models(&d.models, &s.churn)?;

    // Ordre = IA/Sujet_2/src/preprocessing.py::FEATURES
    let x: Vec<f32> = [
        d.age,
        d.tenure_months,
        d.monthly_logins,
        d.weekly_active_days,
        d.avg_session_time,
        d.features_used,
        d.usage_growth_rate,
        d.last_login_days_ago,
        d.monthly_fee,
        d.total_revenue,
        d.payment_failures,
        d.avg_resolution_time,
        d.csat_score,
        d.escalations,
        d.email_open_rate,
        d.marketing_click_rate,
        d.nps_score,
        d.referral_count,
    ]
    .iter()
    .map(|v| *v as f32)
    .chain([
        enc("gender", &d.gender, &[("Female", 0.), ("Male", 1.)])?,
        enc(
            "customer_segment",
            &d.customer_segment,
            &[("Enterprise", 0.), ("Individual", 1.), ("SME", 2.)],
        )?,
        enc(
            "signup_channel",
            &d.signup_channel,
            &[("Mobile", 0.), ("Referral", 1.), ("Web", 2.)],
        )?,
        enc(
            "contract_type",
            &d.contract_type,
            &[("Monthly", 0.), ("Quarterly", 1.), ("Yearly", 2.)],
        )?,
        enc(
            "payment_method",
            &d.payment_method,
            &[("Bank Transfer", 0.), ("Card", 1.), ("PayPal", 2.)],
        )?,
        enc(
            "discount_applied",
            &d.discount_applied,
            &[("No", 0.), ("Yes", 1.)],
        )?,
        enc(
            "price_increase_last_3m",
            &d.price_increase_last_3m,
            &[("No", 0.), ("Yes", 1.)],
        )?,
        enc(
            "survey_response",
            &d.survey_response,
            &[("Neutral", 0.), ("Satisfied", 1.), ("Unsatisfied", 2.)],
        )?,
        enc(
            "complaint_type",
            &d.complaint_type,
            &[
                ("Billing", 0.),
                ("Service", 1.),
                ("Technical", 2.),
                ("Unknown", 3.),
            ],
        )?,
        (d.total_revenue / (d.tenure_months + 1.0)) as f32,
        (d.weekly_active_days * d.avg_session_time) as f32,
        (d.support_tickets * (d.avg_resolution_time + 1.0)) as f32,
    ])
    .collect();

    let mut results = Map::new();
    for name in &names {
        let (prediction, p1) = s.churn.get(name).unwrap().predict_binary(&x)?;
        results.insert(
            name.clone(),
            json!({
                "prediction": prediction,
                "label": if prediction == 1 { "Churn probable" } else { "Client fidèle" },
                "probabilite_churn": round4(p1),
            }),
        );
    }
    Ok(Json(json!({ "results": results })))
}
