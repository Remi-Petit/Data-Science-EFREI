import os
import requests
import streamlit as st

st.set_page_config(layout="wide")

API_URL = os.getenv('API_URL', 'http://localhost:8000')

pg = st.navigation([
    st.Page("pages/sujet1/sujet1.py", title="Sujet 1 – Maintenance prédictive", icon="🔧"),
    st.Page("pages/sujet2/sujet2.py", title="Sujet 2 – Churn client", icon="📉"),
    st.Page("pages/sujet3/sujet3.py", title="Sujet 3 – ROI Marketing", icon="📈"),
    st.Page("pages/modeles/modeles.py", title="Comparaison des modèles", icon="📊"),
])
pg.run()

try:
    r = requests.get(f"{API_URL}/health", timeout=3)
    if r.status_code == 200:
        st.sidebar.success("✅ API opérationnelle")
    else:
        st.sidebar.error("❌ API dégradée")
except Exception:
    st.sidebar.error("❌ API inaccessible")

st.sidebar.link_button("📚 Documentation API", url="/api/docs", use_container_width=True)
st.sidebar.markdown("")
st.sidebar.link_button("🌐 Portfolio", url="https://remipetit.fr/data-science-ia/", use_container_width=True)
st.sidebar.link_button("🐙 GitHub", url="https://github.com/Remi-Petit/Data-Science-EFREI", use_container_width=True)