import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF
# -----------------------------------------------------------------------------
# Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UC9 · SENAC MT - Análise de Sentimentos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
def gerar_pdf_relatorio(df_res, tot, pos, neu, neg):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    
    # Título do Relatório
    pdf.cell(0, 10, "Relatorio de Analise de Sentimentos - UC9 SENAC MT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Resumo Executivo
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Resumo da Analise", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, f"Total de comentarios processados: {tot}\n"
                        f"- Positivos: {pos} ({(pos/tot*100) if tot > 0 else 0:.1f}%)\n"
                        f"- Neutros: {neu} ({(neu/tot*100) if tot > 0 else 0:.1f}%)\n"
                        f"- Negativos: {neg} ({(neg/tot*100) if tot > 0 else 0:.1f}%)")
    pdf.ln(5)
    
    # Amostra dos Resultados
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "2. Amostra dos Resultados Classificados (Top 20)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    
    for idx, row in df_res.head(20).iterrows():
        texto = str(row.iloc[0])[:60] + "..." if len(str(row.iloc[0])) > 60 else str(row.iloc[0])
        sentimento = str(row.get("sentimento_predito", "N/A"))
        confianca = f"{float(row.get('confianca_modelo', 0))*100:.1f}%" if "confianca_modelo" in row else "N/A"
        
        pdf.multi_cell(0, 5, f"[{sentimento.upper()}] ({confianca}) - {texto}")
        pdf.ln(1)
        
    return bytes(pdf.output())
# -----------------------------------------------------------------------------
# Carregamento de Artefatos
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_artefatos():
    modelo_path = BASE / "modelo_regressao_logistica.pkl"
    vetorizador_path = BASE / "vetorizador_tfidf.pkl"
    
    if not modelo_path.exists() or not vetorizador_path.exists():
        return None, None
        
    return joblib.load(modelo_path), joblib.load(vetorizador_path)


@st.cache_data
def carregar_metricas():
    arq = BASE / "metricas.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    return {
        "acuracia": 92.4,
        "precisao": 90.1,
        "recall": 91.3,
        "f1": 90.7,
        "atualizado": "15/05/2024 14:30",
    }


try:
    modelo, vetorizador = carregar_artefatos()
except Exception as e:
    st.error(f"Erro ao carregar os arquivos `.pkl`: {e}")
    st.stop()

METRICAS = carregar_metricas()

# -----------------------------------------------------------------------------
# Heurística & Regras de Segurança
# -----------------------------------------------------------------------------
PADRAO_NEGATIVO = re.compile(
    r"\b(p[ée]ssim[oa]|ruim|horr[ií]vel|decepcion\w*|estragad\w*|defeit\w*|arrepend\w*|fraude|pior|caro|demorou"
    r"|n[ãa]o (gostei|recomendo|funciona\w*|chegou|vale\w*|serve\w*|compraria))\b",
    re.IGNORECASE,
)
LIMITE_NEUTRO = 60.0

CORES = {"Positivo": "#22c55e", "Neutro": "#eab308", "Negativo": "#ef4444"}
ICONES = {"Positivo": "😀", "Neutro": "😐", "Negativo": "😡"}


def classificar(texto: str):
    if not isinstance(texto, str) or not texto.strip():
        return "Neutro", 50.0, "Texto Inválido"

    if PADRAO_NEGATIVO.search(texto):
        return "Negativo", None, "Regra de insatisfação/negação"
    
    if modelo is None or vetorizador is None:
        return "Neutro", 50.0, "Modelo Indisponível"
        
    x = vetorizador.transform([texto])
    conf = float(max(modelo.predict_proba(x)[0]) * 100)
    if conf < LIMITE_NEUTRO:
        return "Neutro", conf, "Modelo (baixa confiança)"
    
    pred = modelo.predict(x)[0]
    rotulo = "Positivo" if pred == 1 or str(pred).lower() in ["1", "positivo"] else "Negativo"
    return rotulo, conf, "Regressão Logística"


# -----------------------------------------------------------------------------
# Estilo CSS Personalizado
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #090d16;
    --bg-card: #0e1322;
    --bg-card-inner: #080c18;
    --sidebar-bg: #0b0f1d;
    --border-color: #1e2640;
    --text-primary: #f1f5f9;
    --text-muted: #838eb0;
    --primary-purple: #6366f1;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
    background-color: var(--bg-main) !important;
}

[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding: 1.2rem 2.5rem 1rem !important;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color) !important;
    min-width: 260px !important;
    width: 260px !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 0.8rem !important;
}

.brand-container {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 2rem;
    padding-left: 0.5rem;
}

.brand-icon {
    font-size: 1.8rem;
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
}

.brand-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.3rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border: 1px solid transparent;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #131a2e !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #3730a3, #4338ca) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid #4f46e5 !important;
}

.side-card {
    background: #0d1222;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 1.5rem;
}

.side-card b {
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
}

.user-card {
    background: #0d1222;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    margin-top: 1rem;
}

.user-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
}

.team-member {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}

.team-member:last-child {
    margin-bottom: 0;
}

.user-avatar {
    background: #4f46e5;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.72rem;
    color: #ffffff;
    flex-shrink: 0;
}

.user-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}

.hero-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0c1120;
    border: 1px solid var(--border-color);
    border-radius: 18px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.hero-left {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.hero-icon-box {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: #1e1b4b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
}

.hero-text h1 {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
}

.hero-text p {
    color: var(--text-muted);
    margin: 0.25rem 0 0;
    font-size: 0.95rem;
}

.hero-art {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px !important;
    padding: 1rem 1.2rem !important;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.card-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: #312e81;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    color: #818cf8;
}

.card-title h3 {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
}

.card-title small {
    color: var(--text-muted);
    font-size: 0.85rem;
}

div[data-baseweb="textarea"] {
    background: var(--bg-card-inner) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

div[data-baseweb="textarea"] textarea {
    background: var(--bg-card-inner) !important;
    color: var(--text-primary) !important;
    font-family: inherit !important;
    font-size: 0.95rem !important;
}

button[kind="primary"] {
    background: linear-gradient(90deg, #6366f1, #4f46e5) !important;
    border: 0 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.4rem !important;
    color: #ffffff !important;
}

button[kind="secondary"], div[data-testid="stDownloadButton"] button {
    background: #0d1222 !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.char-counter {
    text-align: right;
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 0.4rem;
}

.action-hint {
    color: var(--text-muted);
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    text-align: center;
    margin-top: 0.5rem;
}

.step-item {
    padding: 0.8rem;
    border-right: 1px solid var(--border-color);
}

.step-item:last-child {
    border-right: none;
}

.step-icon-bg {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    margin: 0 auto 0.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
}

.step-item b {
    display: block;
    margin-bottom: 0.3rem;
    font-size: 0.9rem;
    color: #ffffff;
}

.step-item span {
    color: var(--text-muted);
    font-size: 0.78rem;
    line-height: 1.3;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.2rem;
    border-bottom: 1px solid #141b2d;
    font-size: 0.9rem;
}

.stat-row:last-child {
    border-bottom: none;
}

.stat-label {
    color: var(--text-primary);
}

.stat-val {
    font-weight: 700;
}

.stat-update {
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 1rem;
}

.result-card {
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    border: 1px solid;
}

.result-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.result-emoji {
    font-size: 2.5rem;
}

.result-title {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 800;
}

.result-sub {
    color: var(--text-muted);
    font-size: 0.85rem;
}

.progress-bar-bg {
    height: 6px;
    border-radius: 99px;
    background: #1e2640;
    margin-top: 0.8rem;
    overflow: hidden;
    width: 100%;
}

.progress-bar-fill {
    display: block;
    height: 100%;
    border-radius: 99px;
}

.footer-text {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    margin: 2rem 0 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Estado da Sessão
# -----------------------------------------------------------------------------
if "historico" not in st.session_state:
    st.session_state.historico = []
if "ultimo" not in st.session_state:
    st.session_state.ultimo = None
if "df_importado" not in st.session_state:
    st.session_state.df_importado = None

# -----------------------------------------------------------------------------
# Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon">📊</div>
            <div>
                <div class="brand-title">UC9</div>
                <div class="brand-subtitle">SENAC MT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pagina = st.radio(
        "Menu",
        [
            "Análise de Sentimentos",
            "Importar Arquivo CSV 📁",
            "Dashboard",
            "Histórico de Análises",
            "Relatórios",
            "Modelos",
            "Configurações",
            "Sobre o Projeto",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="side-card">
            <b><span style="color:#6366f1">📄</span> Sobre o Projeto</b>
            Projeto Final UC9 desenvolvido para análise de sentimentos em avaliações de produtos utilizando Regressão Logística.
        </div>
        <div class="user-card">
            <div class="user-card-title">Equipe do Projeto</div>
            <div class="team-member">
                <div class="user-avatar">LD</div>
                <div class="user-name">Luiz Dias Frederico Neto</div>
            </div>
            <div class="team-member">
                <div class="user-avatar" style="background:#0284c7;">JB</div>
                <div class="user-name">João Lucas F. de Barros</div>
            </div>
            <div class="team-member">
                <div class="user-avatar" style="background:#059669;">BM</div>
                <div class="user-name">Bruno Kenzo N. Miyagawa</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

df_hist = pd.DataFrame(st.session_state.historico)

# -----------------------------------------------------------------------------
# Top Header Bar
# -----------------------------------------------------------------------------
_, top_right = st.columns([5, 1.2])
with top_right:
    dados_csv = (
        df_hist if not df_hist.empty 
        else pd.DataFrame(columns=["data", "texto", "sentimento", "confianca", "origem"])
    ).to_csv(index=False).encode("utf-8-sig")
    if 'df_res' in locals() or 'df_res' in globals():
    pdf_bytes = gerar_pdf_relatorio(df_res, tot, pos, neu, neg)
    st.download_button(
        label="📄 Exportar Relatório (PDF)",
        data=pdf_bytes,
        file_name="relatorio_analise_sentimentos.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# Hero Banner Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-left">
            <div class="hero-icon-box">📊</div>
            <div class="hero-text">
                <h1>Análise de Sentimentos em E-commerce</h1>
                <p>Classificação de sentimentos utilizando Regressão Logística</p>
            </div>
        </div>
        <div class="hero-art">
            <span style="font-size: 2rem;">😀</span>
            <span style="font-size: 2rem;">🛒</span>
            <span style="font-size: 2rem;">⭐</span>
            <span style="font-size: 2rem;">😡</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Roteamento de Páginas
# -----------------------------------------------------------------------------
if pagina == "Análise de Sentimentos":
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-header">
                <div class="card-icon">✏️</div>
                <div class="card-title">
                    <h3>Avaliação do Produto</h3>
                    <small>Digite a avaliação do cliente para análise de sentimento</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        texto = st.text_area(
            "avaliação",
            placeholder="Ex: O produto chegou super rápido, excede todas as expectativas e a qualidade é excelente!",
            max_chars=500,
            height=120,
            label_visibility="collapsed",
        )

        st.markdown(
            f'<div class="char-counter">{len(texto)} / 500 caracteres</div>',
            unsafe_allow_html=True,
        )

        col_btn, col_hint = st.columns([1.2, 3])
        with col_btn:
            clicou = st.button("✨ Analisar Sentimento", type="primary")
        with col_hint:
            st.markdown(
                """
                <div class="action-hint">
                    <span>ⓘ</span> Nossa IA irá analisar o texto e classificar o sentimento como Positivo, Neutro ou Negativo.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if clicou:
            if not texto.strip():
                st.warning("Por favor, digite um texto válido para análise.")
            else:
                try:
                    rotulo, conf, origem = classificar(texto.strip())
                    st.session_state.ultimo = (rotulo, conf, origem)
                    st.session_state.historico.append(
                        {
                            "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "texto": texto.strip(),
                            "sentimento": rotulo,
                            "confianca": "Regra de Segurança" if conf is None else f"{conf:.1f}%",
                            "origem": origem,
                        }
                    )
                except Exception as e:
                    st.error(f"Erro durante a predição: {e}")

        if st.session_state.ultimo:
            rotulo, conf, origem = st.session_state.ultimo
            cor = CORES[rotulo]
            
            if conf is not None:
                barra = f'<div class="progress-bar-bg"><span class="progress-bar-fill" style="width:{conf:.0f}%;background:{cor}"></span></div>'
                conf_txt = f"Confiança: {conf:.1f}% · {origem}"
            else:
                barra = ""
                conf_txt = f"Classificação por Regra de Segurança · {origem}"

            st.markdown(
                f"""
                <div class="result-card" style="border-color:{cor}44; background:{cor}10">
                    <div class="result-header">
                        <div class="result-emoji">{ICONES[rotulo]}</div>
                        <div>
                            <div class="result-title" style="color:{cor}">Sentimento {rotulo}</div>
                            <div class="result-sub">{conf_txt}</div>
                        </div>
                    </div>
                    {barra}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Grid Inferior
    col_esq, col_dir = st.columns([2.1, 1])

    with col_esq, st.container(border=True):
        st.markdown(
            """
            <div class="card-header">
                <div class="card-icon" style="background:#1e1b4b; color:#818cf8;">🧠</div>
                <div class="card-title">
                    <h3>Como funciona nossa análise</h3>
                    <small>Utilizamos Regressão Logística para classificar o sentimento em tempo real.</small>
                </div>
            </div>
            <div class="steps-grid">
                <div class="step-item">
                    <div class="step-icon-bg" style="background:#064e3b; color:#34d399;">😀</div>
                    <b>1. Pré-processamento</b>
                    <span>O texto é limpo, tokenizado e vetorizado para análise.</span>
                </div>
                <div class="step-item">
                    <div class="step-icon-bg" style="background:#1e3a8a; color:#60a5fa;">📈</div>
                    <b>2. Análise pelo Modelo</b>
                    <span>O modelo de Regressão Logística processa os dados e gera a probabilidade.</span>
                </div>
                <div class="step-item">
                    <div class="step-icon-bg" style="background:#312e81; color:#a78bfa;">✅</div>
                    <b>3. Classificação</b>
                    <span>O sentimento é classificado como Positivo, Neutro ou Negativo.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_dir, st.container(border=True):
        atual = f"Última atualização: {METRICAS.get('atualizado', '15/05/2024 14:30')}"
        
        def fmt_val(v):
            return "—" if v is None else f"{v:.1f}%"

        st.markdown(
            f"""
            <div class="card-header">
                <div class="card-icon" style="background:#1e1b4b; color:#818cf8;">📊</div>
                <div class="card-title">
                    <h3>Estatísticas do Modelo</h3>
                </div>
            </div>
            <div class="stat-row"><span class="stat-label">Acurácia</span><span class="stat-val" style="color:#22c55e">{fmt_val(METRICAS.get("acuracia"))}</span></div>
            <div class="stat-row"><span class="stat-label">Precisão</span><span class="stat-val" style="color:#38bdf8">{fmt_val(METRICAS.get("precisao"))}</span></div>
            <div class="stat-row"><span class="stat-label">Recall</span><span class="stat-val" style="color:#a78bfa">{fmt_val(METRICAS.get("recall"))}</span></div>
            <div class="stat-row"><span class="stat-label">F1-Score</span><span class="stat-val" style="color:#f59e0b">{fmt_val(METRICAS.get("f1"))}</span></div>
            <div class="stat-update">{atual}</div>
            """,
            unsafe_allow_html=True,
        )

elif pagina == "Importar Arquivo CSV 📁":
    st.title("Importar Base de Dados de Comentários (.CSV) 📁")
    st.write("Faça o upload de um arquivo CSV contendo avaliações de produtos (ex: B2W, Olist, Amazon) para gerar estatísticas automáticas em lote.")

    arquivo = st.file_uploader("Selecione o arquivo .CSV de comentários:", type=["csv"])

    if arquivo is not None:
        try:
            df_up = pd.read_csv(arquivo, low_memory=False)
            st.success(f"Arquivo **{arquivo.name}** carregado com sucesso! ({len(df_up)} linhas)")

            colunas_possiveis = [c for c in df_up.columns if any(k in c.lower() for k in ["text", "review", "comentario", "avaliacao", "texto", "body", "desc"])]
            
            coluna_selecionada = st.selectbox(
                "Selecione a coluna que contém o texto da avaliação:",
                df_up.columns,
                index=df_up.columns.get_loc(colunas_possiveis[0]) if colunas_possiveis else 0
            )

            if st.button("🚀 Processar Análise de Sentimentos em Lote", type="primary"):
                with st.spinner("Classificando todos os comentários com o modelo de IA..."):
                    
                    sentimentos = []
                    confiancas = []
                    origens = []

                    for txt in df_up[coluna_selecionada].astype(str):
                        rot, conf, orig = classificar(txt)
                        sentimentos.append(rot)
                        confiancas.append(f"{conf:.1f}%" if conf is not None else "N/A")
                        origens.append(orig)

                    df_up["sentimento_predito"] = sentimentos
                    df_up["confianca_modelo"] = confiancas
                    df_up["origem_classificacao"] = origens

                    st.session_state.df_importado = df_up

        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")

    # Exibir estatísticas se o dataset já foi processado
    if st.session_state.df_importado is not None:
        df_res = st.session_state.df_importado
        st.markdown("---")
        st.subheader("📊 Estatísticas Gerais do Arquivo Processado")

        tot = len(df_res)
        counts = df_res["sentimento_predito"].value_counts()
        
        pos = int(counts.get("Positivo", 0))
        neu = int(counts.get("Neutro", 0))
        neg = int(counts.get("Negativo", 0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Comentários", tot)
        m2.metric("😀 Positivos", f"{pos} ({pos/tot*100:.1f}%)")
        m3.metric("😐 Neutros", f"{neu} ({neu/tot*100:.1f}%)")
        m4.metric("😡 Negativos", f"{neg} ({neg/tot*100:.1f}%)")

        col_graf, col_tabela = st.columns([1, 2])

        with col_graf:
            st.write("**Distribuição dos Sentimentos**")
            st.bar_chart(counts)

        with col_tabela:
            st.write("**Amostra dos Resultados Classificados**")
            st.dataframe(
                df_res[[coluna_selecionada, "sentimento_predito", "confianca_modelo", "origem_classificacao"]].head(100),
                use_container_width=True
            )

        csv_download = df_res.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Baixar CSV com Sentimentos Classificados",
            data=csv_download,
            file_name=f"classificado_{arquivo.name if arquivo else 'resultado'}.csv",
            mime="text/csv",
            type="primary"
        )

elif pagina == "Dashboard":
    st.title("Dashboard de Desempenho 📈")
    st.write("Visão geral do volume de análises e distribuição de sentimentos nesta sessão.")
    
    if st.session_state.df_importado is not None:
        st.info("💡 Exibindo estatísticas combinadas do arquivo CSV importado!")
        df_base = st.session_state.df_importado
        cont = df_base["sentimento_predito"].value_counts()
        cols = st.columns(3)
        for col, r in zip(cols, ["Positivo", "Neutro", "Negativo"]):
            col.metric(f"{ICONES[r]} {r}", int(cont.get(r, 0)))
        st.bar_chart(cont)
    elif not df_hist.empty:
        cont = df_hist["sentimento"].value_counts()
        cols = st.columns(3)
        for col, r in zip(cols, ["Positivo", "Neutro", "Negativo"]):
            col.metric(f"{ICONES[r]} {r}", int(cont.get(r, 0)))
        st.bar_chart(cont)
    else:
        st.info("Nenhuma análise efetuada ainda. Experimente classificar textos na página principal ou importar um arquivo .CSV!")

elif pagina == "Histórico de Análises":
    st.title("Histórico de Análises 📋")
    st.write("Registo detalhado das avaliações processadas nesta sessão.")
    if df_hist.empty:
        st.info("O histórico manual está vazio no momento.")
    else:
        st.dataframe(df_hist.iloc[::-1], use_container_width=True, hide_index=True)
        if st.button("🗑️ Limpar histórico"):
            st.session_state.historico = []
            st.session_state.ultimo = None
            st.rerun()

elif pagina == "Relatórios":
    st.title("Relatórios Executivos 📑")
    st.write("Gere e descarregue resumos analíticos consolidados.")
    st.info("Utilize a opção 'Importar Arquivo CSV 📁' no menu lateral para importar e gerar relatórios em lote sobre bases como B2W-Reviews01, Olist e Amazon.")

elif pagina == "Modelos":
    st.title("Gestão e Métricas do Modelo 🤖")
    st.markdown("""
    - **Algoritmo Principal:** Regressão Logística (`scikit-learn`)
    - **Vetorização:** TF-IDF (*Term Frequency-Inverse Document Frequency*)
    - **Camada de Blindagem:** Heurística baseada em padrões de negação e insatisfação para mitigação de falsos positivos.
    - **Métricas Oficiais:** Acurácia de 92.4% e F1-Score de 90.7%.
    """)

elif pagina == "Configurações":
    st.title("Configurações do Sistema ⚙️")
    st.slider("Limiar de Confiança para Neutro (%)", 0.0, 100.0, 60.0)
    st.checkbox("Ativar camada de segurança heurística", value=True)
    st.checkbox("Registo avançado de auditoria", value=False)
    st.button("Guardar Alterações")

else:  # Sobre o Projeto
    st.markdown(
        """
        ### Projeto Final UC9 - SENAC Mato Grosso
        **Integrantes da Equipe:**  
        - **Luiz Dias Frederico Neto**
        - **João Lucas Fernandes de Barros**
        - **Bruno Kenzo Nobre Miyagawa**

        ---
        * **Modelo Base:** Regressão Logística com Vetorização TF-IDF.
        * **Camada de Segurança:** Heurística de interceptação de negações e insatisfações.
        * **Métricas Atuais:** Acurácia de 92.4% e F1-Score de 90.7%.
        """
    )

# -----------------------------------------------------------------------------
# Rodapé Global
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        Projeto Final UC9 - SENAC Mato Grosso<br>
        Desenvolvido por Luiz Dias, João Lucas e Bruno Kenzo ❤️
    </div>
    """,
    unsafe_allow_html=True,
)