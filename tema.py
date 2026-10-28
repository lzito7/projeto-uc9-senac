import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;}
.stApp{background:#0b1020;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.5rem;max-width:1200px;}
section[data-testid="stSidebar"]{background:#0e1429;border-right:1px solid #1f2747;}
.hero{border-bottom:1px solid #1f2747;padding-bottom:1.2rem;margin-bottom:1.5rem;}
.hero h1{font-size:2rem;font-weight:700;margin:0;padding:0;letter-spacing:-.02em;}
.hero p{color:#8a93b8;margin:.4rem 0 0;font-size:1rem;}
div[data-testid="stForm"]{background:#111831;border:1px solid #1f2747;border-radius:12px;}
div[data-testid="stMetric"]{background:#111831;border:1px solid #1f2747;border-radius:10px;padding:.8rem 1rem;}
div[data-baseweb="textarea"],div[data-baseweb="textarea"] textarea{background:#0b1020!important;border-radius:8px!important;}
button[kind="primary"],button[kind="primaryFormSubmit"]{background:#6d4df2;border:0;border-radius:8px;font-weight:600;}
button[data-baseweb="tab"]{font-weight:500;}
</style>
"""


def aplicar_tema():
    st.markdown(CSS, unsafe_allow_html=True)


def renderizar_hero(titulo: str, subtitulo: str):
    st.markdown(f'<div class="hero"><h1>{titulo}</h1><p>{subtitulo}</p></div>', unsafe_allow_html=True)
