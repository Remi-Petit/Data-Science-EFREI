import os
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv('API_URL', 'http://localhost:8000')

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("Navigation")
sujet = st.sidebar.radio(
    "Choisir un sujet",
    options=[
        "Sujet 1 – Maintenance prédictive",
        "Sujet 2 – Churn client",
        "Sujet 3 – ROI Marketing",
    ],
)

st.sidebar.divider()

# ── SUJET 1 ───────────────────────────────────────────────────────────────────

if sujet == "Sujet 1 – Maintenance prédictive":
    S1_MODEL_LABELS = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost",
    }
    CAUSE_LABELS = {
        "bearing":        "Roulement (Bearing)",
        "motor_overheat": "Surchauffe moteur",
        "hydraulic":      "Défaut hydraulique",
        "electrical":     "Défaut électrique",
        "none":           "Aucune panne",
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
        options=list(S1_MODEL_LABELS.keys()),
        default=["random_forest"],
        format_func=lambda x: S1_MODEL_LABELS[x],
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
            response = requests.post(f"{API_URL}/sujet-1/predict", json=payload)
            response.raise_for_status()
            results = response.json()["results"]

            st.divider()
            st.subheader("📊 Résultats par modèle")

            cols = st.columns(len(results))
            for col, (model_key, res) in zip(cols, results.items()):
                with col:
                    st.markdown(f"### {S1_MODEL_LABELS[model_key]}")
                    if res["prediction"] == 1:
                        st.error(f"⚠️ {res['label']}")
                    else:
                        st.success(f"✅ {res['label']}")
                    st.metric("Probabilité de panne", f"{res['probabilite_panne'] * 100:.2f}%")

                    if res["prediction"] == 1 and "cause_potentielle" in res:
                        st.markdown("**Cause potentielle la plus probable :**")
                        cause = res["cause_potentielle"]
                        st.info(f"🔎 {CAUSE_LABELS.get(cause, cause)}")

                        st.markdown("**Détail des probabilités (toutes causes) :**")
                        scores = res["probabilites_causes"]
                        df_causes = pd.DataFrame(
                            {"Cause": [CAUSE_LABELS.get(k, k) for k in scores],
                             "Probabilité (%)": [round(v * 100, 1) for v in scores.values()]}
                        ).sort_values("Probabilité (%)", ascending=False).reset_index(drop=True)
                        st.dataframe(
                            df_causes.style.format({"Probabilité (%)": "{:.1f}%"}),
                            hide_index=True,
                            use_container_width=True,
                        )

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")


# ── SUJET 2 ───────────────────────────────────────────────────────────────────

elif sujet == "Sujet 2 – Churn client":
    S2_MODEL_LABELS = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost",
        "mlp":                 "MLP (Deep Learning)",
    }

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
        options=list(S2_MODEL_LABELS.keys()),
        default=["random_forest"],
        format_func=lambda x: S2_MODEL_LABELS[x],
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
                    st.markdown(f"### {S2_MODEL_LABELS[model_key]}")
                    if res["prediction"] == 1:
                        st.error(f"⚠️ {res['label']}")
                    else:
                        st.success(f"✅ {res['label']}")
                    st.metric("Probabilité de churn", f"{res['probabilite_churn'] * 100:.2f}%")

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")


# ── SUJET 3 ───────────────────────────────────────────────────────────────────

elif sujet == "Sujet 3 – ROI Marketing":
    S3_MODEL_LABELS = {
        "linear_regression": "Régression Linéaire",
        "random_forest":     "Random Forest",
        "xgboost":           "XGBoost",
        "mlp":               "MLP (Deep Learning)",
    }
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

    # ── Inputs budgétaires ────────────────────────────────────────────────────
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

    # ── KPIs de la configuration actuelle ────────────────────────────────────
    total_budget = tv + radio + social_media
    st.divider()
    st.subheader("📊 Indicateurs du scénario")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Budget total", f"{total_budget:.1f} M€")
    k2.metric("Part TV",          f"{tv / total_budget * 100:.1f} %" if total_budget > 0 else "–")
    k3.metric("Part Radio",       f"{radio / total_budget * 100:.1f} %" if total_budget > 0 else "–")
    k4.metric("Part Social Media", f"{social_media / total_budget * 100:.1f} %" if total_budget > 0 else "–")

    # ── Répartition budgétaire (camembert simplifié) ──────────────────────────
    if total_budget > 0:
        budget_df = pd.DataFrame({
            "Canal":  ["TV", "Radio", "Social Media"],
            "Budget": [tv, radio, social_media],
        })
        st.bar_chart(budget_df.set_index("Canal"), height=220)

    st.divider()

    selected_models = st.multiselect(
        "🤖 Modèles à comparer",
        options=list(S3_MODEL_LABELS.keys()),
        default=["linear_regression"],
        format_func=lambda x: S3_MODEL_LABELS[x],
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
                    st.markdown(f"### {S3_MODEL_LABELS[model_key]}")
                    st.metric("Ventes prédites", f"{res['sales_prediction']:.2f} M€")
                    st.metric("ROI estimé", f"{res['roi_estimate']:.2f}x" if res['roi_estimate'] else "–")

                    perf = res["performance"]
                    icon, severity = PERF_COLORS[perf]
                    getattr(st, severity)(f"{icon} Performance : **{perf}**")

            # ── Tableau comparatif ────────────────────────────────────────────
            if len(results) > 1:
                st.divider()
                st.subheader("📋 Tableau comparatif")
                summary = pd.DataFrame([
                    {
                        "Modèle":           S3_MODEL_LABELS[k],
                        "Ventes prédites (M€)": round(v["sales_prediction"], 2),
                        "ROI estimé":       round(v["roi_estimate"], 2) if v["roi_estimate"] else None,
                        "Performance":      v["performance"],
                    }
                    for k, v in results.items()
                ])
                st.dataframe(summary, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")
