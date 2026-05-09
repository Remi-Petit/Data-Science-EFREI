import os
import streamlit as st
import requests

API_URL = os.getenv('API_URL', 'http://localhost:8000')


_MEDALS = ["🥇", "🥈", "🥉"]

@st.cache_data(ttl=300)
def _fetch_models() -> tuple[list[str], dict[str, str]]:
    try:
        r = requests.get(f"{API_URL}/sujet-2/models", timeout=5)
        r.raise_for_status()
        entries = r.json()["models"]
        models = [e["name"] for e in entries]
        labels = {e["name"]: e["label"] for e in entries}
        try:
            sr = requests.get(f"{API_URL}/sujet-2/stats", timeout=5)
            sr.raise_for_status()
            stats = sr.json()["stats"]
            models.sort(key=lambda k: stats.get(k, {}).get("F1-score", 0), reverse=True)
        except Exception:
            pass
        return models, labels
    except Exception:
        fallback = ["logistic_regression", "random_forest", "xgboost", "mlp"]
        return fallback, {k: k for k in fallback}

available_models, model_labels = _fetch_models()

def _label(key: str) -> str:
    idx = available_models.index(key) if key in available_models else -1
    medal = _MEDALS[idx] if 0 <= idx < len(_MEDALS) else ""
    base = model_labels.get(key, key.replace('_', ' ').title())
    return f"{medal} {base}".strip() if medal else base

st.title("📉 Prédiction de churn client")
st.markdown("Renseignez les informations du client pour prédire s'il risque de résilier son abonnement.")
st.divider()

st.subheader("Profil client")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Âge", value=35, step=1, min_value=18, max_value=100)
    tenure_months = st.number_input("Ancienneté (mois)", value=24, step=1, min_value=0)
    monthly_fee = st.number_input("Frais mensuels (€)", value=49.0, step=1.0)
    total_revenue = st.number_input("Revenu total généré (€)", value=1200.0, step=50.0)
    gender = st.selectbox("Genre", options=["Female", "Male"])
    customer_segment = st.selectbox("Segment client", options=["Enterprise", "Individual", "SME"])

with col2:
    monthly_logins = st.number_input("Connexions / mois", value=15, step=1, min_value=0)
    weekly_active_days = st.number_input("Jours actifs / semaine", value=3, step=1, min_value=0, max_value=7)
    avg_session_time = st.number_input("Durée moy. session (min)", value=20.0, step=1.0)
    features_used = st.number_input("Fonctionnalités utilisées", value=5, step=1, min_value=0)
    usage_growth_rate = st.number_input("Taux de croissance d'usage (%)", value=0.05, step=0.01, format="%.2f")
    last_login_days_ago = st.number_input("Dernière connexion (jours)", value=5, step=1, min_value=0)

with col3:
    payment_failures = st.number_input("Échecs de paiement", value=0, step=1, min_value=0)
    support_tickets = st.number_input("Tickets support", value=1, step=1, min_value=0)
    avg_resolution_time = st.number_input("Temps résolution moy. (h)", value=24.0, step=1.0)
    csat_score = st.number_input("Score CSAT", value=4.0, step=0.1, min_value=1.0, max_value=5.0)
    escalations = st.number_input("Escalades", value=0, step=1, min_value=0)
    nps_score = st.number_input("NPS Score", value=7, step=1, min_value=0, max_value=10)

st.subheader("Engagement & marketing")
col4, col5, col6 = st.columns(3)

with col4:
    email_open_rate = st.number_input("Taux d'ouverture email", value=0.30, step=0.01, format="%.2f", min_value=0.0, max_value=1.0)
    marketing_click_rate = st.number_input("Taux de clic marketing", value=0.10, step=0.01, format="%.2f", min_value=0.0, max_value=1.0)
    referral_count = st.number_input("Parrainages", value=0, step=1, min_value=0)

with col5:
    signup_channel = st.selectbox("Canal d'acquisition", options=["Mobile", "Referral", "Web"])
    contract_type = st.selectbox("Type de contrat", options=["Monthly", "Quarterly", "Yearly"])
    payment_method = st.selectbox("Méthode de paiement", options=["Bank Transfer", "Card", "PayPal"])

with col6:
    discount_applied = st.selectbox("Remise appliquée", options=["No", "Yes"])
    price_increase_last_3m = st.selectbox("Hausse de prix (3 derniers mois)", options=["No", "Yes"])
    survey_response = st.selectbox("Réponse enquête", options=["Neutral", "Satisfied", "Unsatisfied"])
    complaint_type = st.selectbox("Type de réclamation", options=["Billing", "Service", "Technical", "Unknown"])

st.divider()

selected_models = st.multiselect(
    "🤖 Modèles à comparer",
    options=available_models,
    default=available_models[:1],
    format_func=_label,
)

st.divider()

if st.button("🔍 Lancer la prédiction", use_container_width=True):
    if not selected_models:
        st.warning("Veuillez sélectionner au moins un modèle.")
        st.stop()

    payload = {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_logins": monthly_logins,
        "weekly_active_days": weekly_active_days,
        "avg_session_time": avg_session_time,
        "features_used": features_used,
        "usage_growth_rate": usage_growth_rate,
        "last_login_days_ago": last_login_days_ago,
        "monthly_fee": monthly_fee,
        "total_revenue": total_revenue,
        "payment_failures": payment_failures,
        "support_tickets": support_tickets,
        "avg_resolution_time": avg_resolution_time,
        "csat_score": csat_score,
        "escalations": escalations,
        "email_open_rate": email_open_rate,
        "marketing_click_rate": marketing_click_rate,
        "nps_score": nps_score,
        "referral_count": referral_count,
        "gender": gender,
        "customer_segment": customer_segment,
        "signup_channel": signup_channel,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "discount_applied": discount_applied,
        "price_increase_last_3m": price_increase_last_3m,
        "survey_response": survey_response,
        "complaint_type": complaint_type,
        "models": selected_models,
    }

    try:
        response = requests.post(f"{API_URL}/sujet-2/predict", json=payload)
        response.raise_for_status()
        results = response.json()["results"]

        st.divider()
        st.subheader("📊 Résultats par modèle")

        cols = st.columns(len(results))
        for col, (model_key, res) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {_label(model_key)}")
                if res["prediction"] == 1:
                    st.error(f"⚠️ {res['label']}")
                else:
                    st.success(f"✅ {res['label']}")
                st.metric("Probabilité de churn", f"{res['probabilite_churn'] * 100:.2f}%")

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
