import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
  font-family: 'Inter', sans-serif;
}
.stApp { background: #070b18; color: #e8ecff; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container {
  padding-top: 1.4rem;
  padding-bottom: 2.5rem;
  max-width: 1180px;
}

section[data-testid="stSidebar"] {
  background: #080c1a;
  border-right: 1px solid #1b2342;
}
section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
  gap: 0.28rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  background: transparent;
  border: 0;
  border-radius: 10px;
  padding: 0.62rem 0.85rem;
  color: #9aa3c7;
  font-weight: 500;
  font-size: 0.92rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #121936;
  color: #e8ecff;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(90deg, #6d4df2 0%, #7a5af5 100%);
  color: #fff;
  font-weight: 600;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.2rem 0.4rem 1rem;
}
.brand-mark {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6 55%, #22c55e);
}
.brand h2 { margin: 0; font-size: 1.15rem; font-weight: 800; letter-spacing: -.03em; color: #fff; }
.brand p { margin: 0; font-size: 0.72rem; color: #8a93b8; letter-spacing: .08em; }

.hero {
  background: linear-gradient(180deg, #12183a 0%, #0e1430 100%);
  border: 1px solid #232c52;
  border-radius: 18px;
  padding: 1.35rem 1.5rem;
  margin-bottom: 1.15rem;
  position: relative;
  overflow: hidden;
}
.hero:after {
  content: "";
  position: absolute;
  right: -40px;
  top: -50px;
  width: 220px;
  height: 220px;
  background: radial-gradient(circle, rgba(109,77,242,.28), transparent 70%);
}
.hero-kicker {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: #6d4df2;
  display: inline-block;
  margin-bottom: 0.7rem;
}
.hero h1 {
  font-size: 1.85rem;
  font-weight: 800;
  margin: 0;
  letter-spacing: -.03em;
  color: #fff;
}
.hero p { color: #9aa3c7; margin: .35rem 0 0; font-size: 0.98rem; }

.card {
  background: #0e1430;
  border: 1px solid #232c52;
  border-radius: 16px;
  padding: 1.2rem 1.3rem 1.3rem;
  margin-bottom: 1rem;
}
.card h3 {
  margin: 0 0 .25rem;
  font-size: 1.05rem;
  font-weight: 700;
  color: #fff;
}
.card .muted { color: #8a93b8; font-size: 0.88rem; margin: 0 0 0.9rem; }

.icon-chip {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: #6d4df2;
  display: inline-block;
  margin-right: 0.55rem;
  vertical-align: middle;
}

div[data-testid="stForm"] {
  background: transparent;
  border: 0;
}
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] textarea {
  background: #070b18 !important;
  border-radius: 12px !important;
  color: #e8ecff !important;
  border: 1px solid #2a335c !important;
}
button[kind="primary"], button[kind="primaryFormSubmit"] {
  background: #6d4df2;
  border: 0;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.45rem 1.1rem;
}
button[kind="secondary"] {
  background: #121936;
  border: 1px solid #2a335c;
  border-radius: 10px;
  color: #e8ecff;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #0e1430;
  border: 1px solid #232c52 !important;
  border-radius: 16px;
  padding: 0.4rem 0.2rem;
}
div[data-testid="stMetric"] {
  background: #0e1430;
  border: 1px solid #232c52;
  border-radius: 12px;
  padding: .85rem 1rem;
}
div[data-testid="stMetric"] label { color: #8a93b8; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff; }

.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 0.55rem 0;
  border-bottom: 1px solid #1b2342;
  font-size: 0.92rem;
}
.stat-row:last-child { border-bottom: 0; }
.stat-ok { color: #34d399; font-weight: 700; }
.stat-warn { color: #fbbf24; font-weight: 700; }

.steps {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.8rem;
}
.step { text-align: center; padding: 0.4rem 0.5rem; }
.step h4 { margin: 0.45rem 0 0.25rem; font-size: 0.92rem; color: #fff; }
.step p { margin: 0; color: #8a93b8; font-size: 0.8rem; }
.dot {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  margin: 0 auto;
  border: 2px solid #2a335c;
}
.dot-g { background: radial-gradient(circle at 30% 30%, #4ade80, #166534); }
.dot-p { background: radial-gradient(circle at 30% 30%, #818cf8, #4c1d95); }
.dot-v { background: radial-gradient(circle at 30% 30%, #c084fc, #6d28d9); }

.result-pos, .result-neg {
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin: 0.6rem 0 1rem;
}
.result-pos { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.35); }
.result-neg { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.35); }
.result-pos h2, .result-neg h2 { margin: 0 0 .2rem; font-size: 1.2rem; }
.result-pos h2 { color: #34d399; }
.result-neg h2 { color: #f87171; }

.footer-note {
  text-align: center;
  color: #6b7394;
  font-size: 0.82rem;
  margin-top: 1.6rem;
  padding-top: 0.8rem;
  border-top: 1px solid #1b2342;
}
.user-card {
  background: #0e1430;
  border: 1px solid #232c52;
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
  margin-top: 1rem;
}
.user-card strong { color: #fff; display: block; }
.user-card span { color: #8a93b8; font-size: 0.78rem; }
.about-mini {
  background: #0e1430;
  border: 1px solid #232c52;
  border-radius: 12px;
  padding: 0.9rem;
  color: #9aa3c7;
  font-size: 0.82rem;
  margin-top: 1.2rem;
}

@media (max-width: 900px) {
  .steps { grid-template-columns: 1fr; }
  .hero h1 { font-size: 1.4rem; }
}
</style>
"""


def aplicar_tema():
    st.markdown(CSS, unsafe_allow_html=True)


def renderizar_hero(titulo: str, subtitulo: str):
    st.markdown(
        f'<div class="hero"><span class="hero-kicker"></span>'
        f"<h1>{titulo}</h1><p>{subtitulo}</p></div>",
        unsafe_allow_html=True,
    )


def card_abrir(titulo: str, subtitulo: str = "") -> None:
    extra = f'<p class="muted">{subtitulo}</p>' if subtitulo else ""
    st.markdown(
        f'<div class="card"><h3><span class="icon-chip"></span>{titulo}</h3>{extra}',
        unsafe_allow_html=True,
    )


def card_fechar() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
