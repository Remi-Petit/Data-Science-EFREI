import os
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

_MEDALS = ["🥇", "🥈", "🥉"]


# ── Fetch ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch(subject: str) -> tuple[dict, dict]:
    labels_r = requests.get(f"{API_URL}/sujet-{subject}/models", timeout=5)
    stats_r  = requests.get(f"{API_URL}/sujet-{subject}/stats",  timeout=5)
    labels_r.raise_for_status()
    stats_r.raise_for_status()
    labels = {e["name"]: e["label"] for e in labels_r.json()["models"]}
    return labels, stats_r.json()["stats"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_df(group: dict, labels: dict, rank_col: str, ascending: bool) -> pd.DataFrame:
    rows = [{"_key": k, "Modèle": labels.get(k, k), **metrics} for k, metrics in group.items()]
    df = (
        pd.DataFrame(rows)
        .sort_values(rank_col, ascending=ascending)
        .reset_index(drop=True)
    )
    df.insert(0, "Rang", [_MEDALS[i] if i < len(_MEDALS) else f"#{i + 1}" for i in range(len(df))])
    return df


def _show_group(
    group: dict,
    labels: dict,
    rank_col: str,
    ascending: bool = False,
    as_pct: bool = True,
):
    if not group:
        st.warning("Aucune statistique disponible pour ce groupe.")
        return

    df = _build_df(group, labels, rank_col, ascending)
    metric_cols = [c for c in df.columns if c not in ("Rang", "Modèle", "_key")]
    best = df.iloc[0]

    # ── Meilleur modèle ──────────────────────────────────────────────────────
    st.markdown(f"#### 🏆 Meilleur modèle : **{best['Modèle']}**")
    m_cols = st.columns(len(metric_cols))
    for i, col in enumerate(metric_cols):
        val = best[col]
        fmt = f"{val:.2%}" if as_pct else f"{val:.4f}"
        m_cols[i].metric(col, fmt)

    st.divider()

    # ── Tableau classé ───────────────────────────────────────────────────────
    display = df[["Rang", "Modèle"] + metric_cols].copy()
    for col in metric_cols:
        display[col] = display[col].apply(
            lambda x: f"{x:.2%}" if as_pct else f"{x:.4f}"
        )
    st.dataframe(display, hide_index=True, use_container_width=True)

    # ── Graphique ────────────────────────────────────────────────────────────
    chart_df = (
        df[["Modèle", rank_col]]
        .set_index("Modèle")
        .sort_values(rank_col, ascending=True)   # bar_chart affiche bas → haut
    )
    st.markdown(f"**{rank_col}** par modèle")
    st.bar_chart(chart_df, horizontal=True)


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("📊 Comparaison des modèles")
st.markdown(
    "Classement des modèles entraînés pour chaque sujet, du meilleur au moins performant."
)
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔧 Sujet 1 – Maintenance",
    "📉 Sujet 2 – Churn",
    "📈 Sujet 3 – Marketing",
])

# ── Sujet 1 ──────────────────────────────────────────────────────────────────
with tab1:
    try:
        labels1, stats1 = _fetch("1")
        sub_a, sub_b = st.tabs(["Détection panne 24h", "Type de panne"])
        with sub_a:
            _show_group(stats1.get("failure_24h", {}), labels1, "F1-score")
        with sub_b:
            _show_group(stats1.get("failure_type", {}), labels1, "F1-score")
    except Exception as e:
        st.error(f"Impossible de charger les statistiques Sujet 1 : {e}")

# ── Sujet 2 ──────────────────────────────────────────────────────────────────
with tab2:
    try:
        labels2, stats2 = _fetch("2")
        _show_group(stats2, labels2, "F1-score")
    except Exception as e:
        st.error(f"Impossible de charger les statistiques Sujet 2 : {e}")

# ── Sujet 3 ──────────────────────────────────────────────────────────────────
with tab3:
    try:
        labels3, stats3 = _fetch("3")
        sub_reg, sub_cls = st.tabs(["Régression (Ventes)", "Classification (Performance campagne)"])
        with sub_reg:
            _show_group(
                stats3.get("regression", {}), labels3,
                rank_col="R²", ascending=False, as_pct=False,
            )
        with sub_cls:
            _show_group(
                stats3.get("classification", {}), labels3,
                rank_col="F1 (macro)", ascending=False, as_pct=True,
            )
    except Exception as e:
        st.error(f"Impossible de charger les statistiques Sujet 3 : {e}")
