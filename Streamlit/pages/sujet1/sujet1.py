import os
import json
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv('API_URL', 'http://localhost:8000')

_MACHINES_FILE = os.path.join(os.path.dirname(__file__), 'machines.json')
_MEDALS = ["🥇", "🥈", "🥉"]

@st.cache_data(ttl=300)
def _fetch_models() -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """`models_24h`, `labels`, `groups` — groups = {group_name: [sorted model keys]}."""
    _fallback = ["logistic_regression", "random_forest", "xgboost"]
    try:
        r = requests.get(f"{API_URL}/sujet-1/models", timeout=5)
        r.raise_for_status()
        raw = r.json()["models"]                          # dict {group: [{name, label}]}
        labels = {}
        groups = {}
        for gname, entries in raw.items():
            groups[gname] = [e["name"] for e in entries]  # already sorted by API
            for e in entries:
                labels[e["name"]] = e["label"]
        models_24h = groups.get("failure_24h", _fallback)
        return models_24h, labels, groups
    except Exception:
        return _fallback, {k: k for k in _fallback}, {}


@st.cache_data(ttl=300)
def _load_machines() -> list[dict]:
    with open(_MACHINES_FILE, encoding='utf-8') as f:
        return json.load(f)

available_models, model_labels, _all_groups = _fetch_models()

def _label(key: str, sorted_list: list[str] | None = None) -> str:
    lst = sorted_list if sorted_list is not None else available_models
    idx = lst.index(key) if key in lst else -1
    medal = _MEDALS[idx] if 0 <= idx < len(_MEDALS) else ""
    base = model_labels.get(key, key.replace('_', ' ').title())
    return f"{medal} {base}".strip() if medal else base

@st.cache_data(ttl=300)
def _fetch_cause_labels() -> dict[str, str]:
    try:
        r = requests.get(f"{API_URL}/sujet-1/cause-labels", timeout=5)
        r.raise_for_status()
        return r.json()["labels"]
    except Exception:
        return {}

CAUSE_LABELS = _fetch_cause_labels()

st.title("🔧 Prédiction de panne machine")
st.divider()

tab_pred, tab_dash = st.tabs(["🔍 Prédiction", "📟 Dashboard machines"])

# ── TAB 1 : Prédiction ────────────────────────────────────────────────────────
with tab_pred:
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
                    st.markdown(f"### {_label(model_key)}")
                    if res["prediction"] == 1:
                        st.error(f"⚠️ {res['label']}")
                    else:
                        st.success(f"✅ {res['label']}")
                    st.metric("Probabilité de panne", f"{res['probabilite_panne'] * 100:.2f}%")
                    if "rul_heures" in res:
                        st.metric("⏱️ Durée de vie restante", f"{res['rul_heures']:.1f} h")

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

# ── TAB 2 : Dashboard machines ────────────────────────────────────────────────
with tab_dash:
    st.markdown("État en temps réel des 10 machines de l'usine.")
    st.divider()

    machines = _load_machines()

    models_cause = _all_groups.get("failure_type", available_models)
    models_rul   = _all_groups.get("rul", available_models)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        selected_model = st.selectbox(
            "🤖 Détection de panne",
            options=available_models, index=0,
            format_func=lambda k: _label(k, available_models),
        )
    with col_s2:
        selected_cause = st.selectbox(
            "🔎 Cause de panne",
            options=models_cause, index=0,
            format_func=lambda k: _label(k, models_cause),
        )
    with col_s3:
        selected_rul = st.selectbox(
            "⏱️ Durée de vie restante",
            options=models_rul, index=0,
            format_func=lambda k: _label(k, models_rul),
        )

    if st.button("🔄 Analyser toutes les machines", use_container_width=True):
        results_all = []
        prog = st.progress(0, text="Analyse en cours…")
        for i, m in enumerate(machines):
            payload = {
                "vibration_rms":          m["vibration_rms"],
                "temperature_motor":      m["temperature_motor"],
                "current_phase_avg":      m["current_phase_avg"],
                "pressure_level":         m["pressure_level"],
                "rpm":                    m["rpm"],
                "hours_since_maintenance": m["hours_since_maintenance"],
                "ambient_temp":           m["ambient_temp"],
                "machine_type_enc":       m["type"],
                "operating_mode_enc":     m["operating_mode_enc"],
                "hour":                   m["hour"],
                "dayofweek":              m["dayofweek"],
                "month":                  m["month"],
                "models":                 [selected_model],
                "model_cause":            selected_cause,
                "model_rul":              selected_rul,
            }
            try:
                r = requests.post(f"{API_URL}/sujet-1/predict", json=payload, timeout=10)
                r.raise_for_status()
                res = r.json()["results"][selected_model]
                results_all.append({**m, **res})
            except Exception as e:
                results_all.append({**m, "prediction": -1, "probabilite_panne": 0.0, "label": f"Erreur: {e}"})
            prog.progress((i + 1) / len(machines), text=f"Analyse en cours… {i + 1}/{len(machines)}")
        prog.empty()
        st.session_state["dashboard_results"] = results_all

    # Affichage des résultats
    if "dashboard_results" in st.session_state:
        results_all = st.session_state["dashboard_results"]

        # KPIs globaux
        n_ok      = sum(1 for r in results_all if r.get("prediction") == 0)
        n_risk    = sum(1 for r in results_all if r.get("prediction") == 1)
        n_err     = sum(1 for r in results_all if r.get("prediction") == -1)
        avg_prob  = sum(r.get("probabilite_panne", 0) for r in results_all) / len(results_all)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Machines analysées", len(results_all))
        k2.metric("✅ En bon état",      n_ok,   delta=None)
        k3.metric("⚠️ À risque",         n_risk, delta=None)
        k4.metric("Prob. panne moy.",   f"{avg_prob * 100:.1f}%")

        st.divider()

        # Grille de cartes
        cols = st.columns(2)
        for i, r in enumerate(results_all):
            with cols[i % 2]:
                pred = r.get("prediction", -1)
                prob = r.get("probabilite_panne", 0)
                icon = "✅" if pred == 0 else ("⚠️" if pred == 1 else "❓")
                color = "success" if pred == 0 else ("error" if pred == 1 else "warning")

                with st.container(border=True):
                    hcol, pcol = st.columns([3, 1])
                    with hcol:
                        st.markdown(f"**{icon} {r['id']} – {r['name']}**")
                        st.caption(f"{r['type_label']} · {r['location']}")
                    with pcol:
                        getattr(st, color)(f"{prob * 100:.0f}%")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Vibration",    f"{r['vibration_rms']} g")
                    mc2.metric("Température",  f"{r['temperature_motor']} °C")
                    mc3.metric("Maintenance",  f"{r['hours_since_maintenance']} h")

                    if "rul_heures" in r:
                        rul_val = r["rul_heures"]
                        rul_color = "normal" if rul_val > 48 else ("off" if rul_val > 12 else "inverse")
                        st.metric("⏱️ Durée de vie restante", f"{rul_val:.1f} h")

                    if pred == 1 and r.get("cause_potentielle"):
                        st.info(f"🔎 Cause probable : {CAUSE_LABELS.get(r['cause_potentielle'], r['cause_potentielle'])}")
                        if r.get("probabilites_causes"):
                            with st.expander("Détail des probabilités par cause"):
                                scores = r["probabilites_causes"]
                                df_causes = pd.DataFrame(
                                    {"Cause": [CAUSE_LABELS.get(k, k) for k in scores],
                                     "Probabilité": [round(v * 100, 1) for v in scores.values()]}
                                ).sort_values("Probabilité", ascending=False).reset_index(drop=True)
                                st.dataframe(
                                    df_causes.style.format({"Probabilité": "{:.1f}%"}),
                                    hide_index=True,
                                    use_container_width=True,
                                )
    else:
        # Aperçu statique des machines sans prédiction
        st.info("Cliquez sur **Analyser toutes les machines** pour lancer les prédictions.")
        rows = [{"ID": m["id"], "Nom": m["name"], "Type": m["type_label"], "Localisation": m["location"],
                 "Vibration (g)": m["vibration_rms"], "Temp. moteur (°C)": m["temperature_motor"],
                 "H. depuis maint.": m["hours_since_maintenance"]} for m in machines]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

