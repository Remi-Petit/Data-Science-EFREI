import os
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv('API_URL', 'http://localhost:8000')


@st.cache_data(ttl=60)
def _fetch_models() -> tuple[list[str], dict[str, str]]:
    try:
        r = requests.get(f"{API_URL}/sujet-3/models", timeout=5)
        r.raise_for_status()
        entries = r.json()["models"]
        models = [e["name"] for e in entries]
        labels = {e["name"]: e["label"] for e in entries}
        return models, labels
    except Exception:
        fallback = ["linear_regression", "random_forest", "xgboost", "mlp"]
        return fallback, {k: k for k in fallback}

available_models, model_labels = _fetch_models()

def _label(key: str) -> str:
    return model_labels.get(key, key.replace('_', ' ').title())

PERF_COLORS = {
    "Low":    ("🔴", "error"),
    "Medium": ("🟡", "warning"),
    "High":   ("🟢", "success"),
}

st.title("📈 Optimisation du ROI Marketing")
st.markdown(
    "Simulez l'impact d'un mix média sur les ventes et estimez le ROI "
    "de votre campagne en temps réel."
)
st.divider()

st.subheader("💰 Budget média")
col1, col2, col3, col4 = st.columns(4)
with col1:
    tv = st.number_input("Budget TV (M€)", value=50.0, min_value=0.0, step=1.0)
with col2:
    radio = st.number_input("Budget Radio (M€)", value=18.0, min_value=0.0, step=0.5)
with col3:
    social_media = st.number_input("Budget Social Media (M€)", value=3.0, min_value=0.0, step=0.1)
with col4:
    influencer = st.selectbox("Type d'influenceur", options=["Macro", "Mega", "Micro", "Nano"])

total_budget = tv + radio + social_media
st.divider()
st.subheader("📊 Indicateurs du scénario")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Budget total", f"{total_budget:.1f} M€")
k2.metric("Part TV",           f"{tv / total_budget * 100:.1f} %" if total_budget > 0 else "–")
k3.metric("Part Radio",        f"{radio / total_budget * 100:.1f} %" if total_budget > 0 else "–")
k4.metric("Part Social Media", f"{social_media / total_budget * 100:.1f} %" if total_budget > 0 else "–")

if total_budget > 0:
    budget_df = pd.DataFrame({
        "Canal":  ["TV", "Radio", "Social Media"],
        "Budget": [tv, radio, social_media],
    })
    st.bar_chart(budget_df.set_index("Canal"), height=220)

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
    if total_budget <= 0:
        st.warning("Le budget total doit être supérieur à 0.")
        st.stop()

    payload = {
        "tv":           tv,
        "radio":        radio,
        "social_media": social_media,
        "influencer":   influencer,
        "models":       selected_models,
    }

    try:
        response = requests.post(f"{API_URL}/sujet-3/predict", json=payload)
        response.raise_for_status()
        results = response.json()["results"]

        st.divider()
        st.subheader("📊 Résultats par modèle")

        cols = st.columns(len(results))
        for col, (model_key, res) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {_label(model_key)}")
                st.metric("Ventes prédites", f"{res['sales_prediction']:.2f} M€")
                st.metric("ROI estimé", f"{res['roi_estimate']:.2f}x" if res['roi_estimate'] else "–")

                perf = res["performance"]
                icon, severity = PERF_COLORS[perf]
                getattr(st, severity)(f"{icon} Performance : **{perf}**")

        if len(results) > 1:
            st.divider()
            st.subheader("📋 Tableau comparatif")
            summary = pd.DataFrame([
                {
                    "Modèle":                _label(k),
                    "Ventes prédites (M€)":  round(v["sales_prediction"], 2),
                    "ROI estimé":            round(v["roi_estimate"], 2) if v["roi_estimate"] else None,
                    "Performance":           v["performance"],
                }
                for k, v in results.items()
            ])
            st.dataframe(summary, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
