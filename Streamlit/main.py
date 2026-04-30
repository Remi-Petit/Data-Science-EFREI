import os
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv('API_URL', 'http://localhost:8000')

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}

CAUSE_LABELS = {
    "bearing":       "Roulement (Bearing)",
    "motor_overheat": "Surchauffe moteur",
    "hydraulic":     "Défaut hydraulique",
    "electrical":    "Défaut électrique",
}

st.title("🔧 Prédiction de panne machine")
st.markdown("Renseignez les paramètres de la machine pour prédire un risque de panne dans les 24h.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    vibration_rms = st.number_input("Vibration RMS", value=2.0, step=0.1)
    temperature_motor = st.number_input("Température moteur (°C)", value=70.0, step=0.5)
    current_phase_avg = st.number_input("Courant moyen (A)", value=10.0, step=0.1)
    ambient_temp = st.number_input("Température ambiante (°C)", value=25.0, step=0.5)

with col2:
    pressure_level = st.number_input("Pression (bar)", value=3.0, step=0.1)
    rpm = st.number_input("RPM", value=1450.0, step=10.0)
    hours_since_maintenance = st.number_input("Heures depuis maintenance", value=200.0, step=10.0)

with col3:
    machine_type_enc = st.selectbox("Type de machine", options=[0, 1, 2], format_func=lambda x: ["Pump", "Compressor", "Motor"][x])
    operating_mode_enc = st.selectbox("Mode opératoire", options=[0, 1, 2], format_func=lambda x: ["normal", "idle", "peak"][x])
    hour = st.slider("Heure", 0, 23, 12)
    dayofweek = st.slider("Jour de la semaine", 0, 6, 1)
    month = st.slider("Mois", 1, 12, 6)

st.divider()

selected_models = st.multiselect(
    "🤖 Modèles à comparer",
    options=list(MODEL_LABELS.keys()),
    default=["random_forest"],
    format_func=lambda x: MODEL_LABELS[x],
)

st.divider()

if st.button("🔍 Lancer la prédiction", use_container_width=True):
    if not selected_models:
        st.warning("Veuillez sélectionner au moins un modèle.")
        st.stop()

    payload = {
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm,
        "hours_since_maintenance": hours_since_maintenance,
        "ambient_temp": ambient_temp,
        "machine_type_enc": machine_type_enc,
        "operating_mode_enc": operating_mode_enc,
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "models": selected_models,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        results = response.json()["results"]

        st.divider()
        st.subheader("📊 Résultats par modèle")

        cols = st.columns(len(results))
        for col, (model_key, res) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {MODEL_LABELS[model_key]}")
                if res["prediction"] == 1:
                    st.error(f"⚠️ {res['label']}")
                else:
                    st.success(f"✅ {res['label']}")
                st.metric("Probabilité de panne", f"{res['probabilite_panne'] * 100:.2f}%")

                if res["prediction"] == 1 and "cause_potentielle" in res:
                    st.markdown("**Cause potentielle :**")
                    cause = res["cause_potentielle"]
                    st.info(f"🔎 {CAUSE_LABELS.get(cause, cause)}")

                    st.markdown("**Probabilités par cause :**")
                    scores = res["probabilites_causes"]
                    df_causes = pd.DataFrame(
                        {"Cause": [CAUSE_LABELS.get(k, k) for k in scores],
                         "Probabilité": [v * 100 for v in scores.values()]}
                    ).sort_values("Probabilité", ascending=False).reset_index(drop=True)
                    st.dataframe(
                        df_causes.style.format({"Probabilité": "{:.1f}%"}),
                        hide_index=True,
                        use_container_width=True,
                    )

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
