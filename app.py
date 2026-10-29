"""
================================================================================
PROJETO FINAL UC9 - ANÁLISE DE SENTIMENTOS EM E-COMMERCE (SENAC-MT)
================================================================================
Equipe Acadêmica:
- Luiz Dias Frederico Neto
- Bruno Kenzo Nobre Miyagawa
- João Lucas Fernandes de Barros
================================================================================
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import json
from tema import aplicar_tema, renderizar_hero
from preprocessamento import normalizar_texto
import pandas as pd
import streamlit as st

try:
    from fpdf import FPDF
    FPDF_INSTALADO = True
except ImportError:
    FPDF = object  # type: ignore
    FPDF_INSTALADO = False


# ==============================================================================
# 1. CONFIGURAÇÕES GLOBAIS E CONSTANTES
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
CAMINHO_MODELO = BASE_DIR / "models" / "modelo_regressao_logistica.pkl"
CAMINHO_VETORIZADOR = BASE_DIR / "models" / "vetorizador_tfidf.pkl"
CAMINHO_METRICAS = BASE_DIR / "models" / "metricas.json"
CAMINHO_B2W_RESUMO = BASE_DIR / "data" / "b2w_resumo.json"
CAMINHO_B2W_AMOSTRA = BASE_DIR / "data" / "b2w_amostra.csv"
LIMITE_CARACTERES = 500

PAGINAS = [
    "Análise de Sentimentos",
    "Dashboard",
    "Histórico de Análises",
    "Relatórios",
    "Modelos",
    "Configurações",
    "Sobre o Projeto",
]

TERMOS_NEGATIVOS = [
    "péssimo", "pessimo", "ruim", "horrível", "horrivel",
    "decepcionou", "estragado", "defeituoso", "caro", "arrependi",
    "demorou", "fraude", "péssima", "pessima", "defeito", "pior",
    "estragou", "danificado", "falha", "lento", "descartável",
    "não gostei", "nao gostei", "não recomendo", "nao recomendo", "não funciona", "nao funciona",
    "não chegou", "nao chegou", "não vale", "nao vale"
]

PADRAO_NEGACAO_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(termo) for termo in TERMOS_NEGATIVOS) + r")\b",
    re.IGNORECASE
)

st.set_page_config(
    page_title="Análise de Sentimentos | SENAC-MT",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)
aplicar_tema()


# ==============================================================================
# 2. CAMADA DE CARREGAMENTO DE ARTEFATOS DE ML (COM CACHE E VALIDAÇÃO)
# ==============================================================================
@st.cache_resource(show_spinner="Carregando modelos de Machine Learning...")
def carregar_artefatos(caminho_mod: Path, caminho_vet: Path) -> Tuple[Any, Any]:
    if not caminho_mod.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: '{caminho_mod.name}'.")
    if not caminho_vet.exists():
        raise FileNotFoundError(f"Arquivo do vetorizador não encontrado: '{caminho_vet.name}'.")

    modelo = joblib.load(caminho_mod)
    vetorizador = joblib.load(caminho_vet)

    if not hasattr(modelo, "predict") or not hasattr(modelo, "predict_proba"):
        raise TypeError("O objeto do modelo carregado não possui os métodos 'predict' e 'predict_proba'.")
    if not hasattr(vetorizador, "transform"):
        raise TypeError("O vetorizador carregado não possui o método 'transform'.")

    return modelo, vetorizador


@st.cache_data(show_spinner=False)
def carregar_metricas_oficiais() -> Optional[Dict[str, Any]]:
    if not CAMINHO_METRICAS.exists():
        return None
    return json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def carregar_resumo_b2w() -> Optional[Dict[str, Any]]:
    if not CAMINHO_B2W_RESUMO.exists():
        return None
    return json.loads(CAMINHO_B2W_RESUMO.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def carregar_amostra_b2w() -> Optional[pd.DataFrame]:
    if not CAMINHO_B2W_AMOSTRA.exists():
        return None
    return pd.read_csv(CAMINHO_B2W_AMOSTRA)


# ==============================================================================
# 3. CAMADA DE LÓGICA DE NEGÓCIO E INFERÊNCIA NLP
# ==============================================================================
def verificar_heuristica_insatisfacao(texto: str) -> Tuple[bool, List[str]]:
    if not texto:
        return False, []
    encontrados = PADRAO_NEGACAO_REGEX.findall(texto.lower())
    termos_unicos = list(dict.fromkeys(encontrados))
    return len(termos_unicos) > 0, termos_unicos


def analisar_texto_individual(
    texto: str,
    modelo: Any,
    vetorizador: Any,
    usar_heuristica: bool = False
) -> Dict[str, Any]:
    texto_limpo = texto.strip()
    if not texto_limpo:
        raise ValueError("O texto fornecido está vazio.")

    heuristica_acionada, termos_encontrados = (
        verificar_heuristica_insatisfacao(texto_limpo) if usar_heuristica else (False, [])
    )

    vetor = vetorizador.transform([normalizar_texto(texto_limpo)])
    predicao_ml = int(modelo.predict(vetor)[0])
    probabilidades = modelo.predict_proba(vetor)[0]

    prob_negativo = float(probabilidades[0])
    prob_positivo = float(probabilidades[1]) if len(probabilidades) > 1 else (1.0 - prob_negativo)

    if heuristica_acionada:
        sentimento_final = "Negativo"
        classe_final = 0
        confianca_final = prob_negativo * 100.0
        motivo = f"Classificação crítica por heurística de negação/insatisfação (Termos: {', '.join(termos_encontrados)})"
    else:
        classe_final = predicao_ml
        sentimento_final = "Positivo" if predicao_ml == 1 else "Negativo"
        confianca_final = max(probabilidades) * 100.0
        motivo = "Predição pura por Regressão Logística TF-IDF"

    return {
        "texto": texto_limpo,
        "classe": classe_final,
        "sentimento": sentimento_final,
        "confianca": confianca_final,
        "prob_positivo": prob_positivo * 100.0,
        "prob_negativo": prob_negativo * 100.0,
        "heuristica_acionada": heuristica_acionada,
        "termos_detectados": termos_encontrados,
        "motivo": motivo,
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


def analisar_dataframe_lote(
    df: pd.DataFrame,
    coluna_texto: str,
    modelo: Any,
    vetorizador: Any,
    usar_heuristica: bool = False
) -> pd.DataFrame:
    if coluna_texto not in df.columns:
        raise KeyError(f"A coluna '{coluna_texto}' não foi encontrada no DataFrame.")

    df_proc = df.copy()
    series_texto = df_proc[coluna_texto].fillna("").astype(str).str.strip()

    X_tfidf = vetorizador.transform(series_texto.map(normalizar_texto))
    predicoes_ml = modelo.predict(X_tfidf)
    probabilidades = modelo.predict_proba(X_tfidf)

    prob_neg = probabilidades[:, 0]
    prob_pos = probabilidades[:, 1] if probabilidades.shape[1] > 1 else (1.0 - prob_neg)
    confiancas = probabilidades.max(axis=1) * 100.0

    sentimentos = ["Positivo" if p == 1 else "Negativo" for p in predicoes_ml]
    heuristica_flags = [False] * len(df_proc)
    termos_lista = [""] * len(df_proc)

    if usar_heuristica:
        for idx, txt in enumerate(series_texto):
            acionou, termos = verificar_heuristica_insatisfacao(txt)
            if acionou:
                heuristica_flags[idx] = True
                termos_lista[idx] = ", ".join(termos)
                sentimentos[idx] = "Negativo"
                confiancas[idx] = prob_neg[idx] * 100.0

    df_proc["sentimento_predito"] = sentimentos
    df_proc["confianca_pct"] = confiancas.round(2)
    df_proc["prob_positivo_pct"] = (prob_pos * 100.0).round(2)
    df_proc["prob_negativo_pct"] = (prob_neg * 100.0).round(2)
    df_proc["heuristica_seguranca"] = heuristica_flags
    df_proc["termos_heuristica"] = termos_lista

    return df_proc


def calcular_metricas_resumo(df_resultados: pd.DataFrame) -> Dict[str, Any]:
    total = len(df_resultados)
    if total == 0:
        return {
            "total": 0, "positivos": 0, "negativos": 0,
            "pct_positivos": 0.0, "pct_negativos": 0.0,
            "confianca_media": 0.0, "total_heuristicas": 0
        }

    positivos = int((df_resultados["sentimento_predito"] == "Positivo").sum())
    negativos = int((df_resultados["sentimento_predito"] == "Negativo").sum())
    heuristicas = int(df_resultados["heuristica_seguranca"].sum()) if "heuristica_seguranca" in df_resultados else 0
    conf_media = float(df_resultados["confianca_pct"].mean())

    return {
        "total": total,
        "positivos": positivos,
        "negativos": negativos,
        "pct_positivos": round((positivos / total) * 100.0, 2),
        "pct_negativos": round((negativos / total) * 100.0, 2),
        "confianca_media": round(conf_media, 2),
        "total_heuristicas": heuristicas
    }


# ==============================================================================
# 4. CAMADA DE GERAÇÃO DE RELATÓRIO EXECUTIVO EM PDF (fpdf2)
# ==============================================================================
def sanitizar_texto_latin1(texto: Any) -> str:
    if texto is None:
        return ""
    texto_str = str(texto)

    substituicoes = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...", "\xa0": " ",
        "•": "-", "–": "-", "—": "-",
        "“": '"', "”": '"', "’": "'", "‘": "'"
    }
    for orig, subst in substituicoes.items():
        texto_str = texto_str.replace(orig, subst)

    return texto_str.encode("latin-1", "replace").decode("latin-1")


if FPDF_INSTALADO:
    class RelatorioExecutivoPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(15, 34, 64)
            self.cell(0, 7, sanitizar_texto_latin1("SENAC-MT | Faculdade de Tecnologia SENAC"), ln=True, align="L")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, sanitizar_texto_latin1("Projeto Final UC9 - Inteligência Artificial & Machine Learning"), ln=True, align="L")
            self.set_draw_color(200, 205, 215)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y() + 2, self.w - self.r_margin, self.get_y() + 2)
            self.ln(6)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 130)
            data_str = datetime.now().strftime("%d/%m/%Y às %H:%M")
            self.cell(0, 10, sanitizar_texto_latin1(f"Emitido em: {data_str} | Página {self.page_no()}/{{nb}}"), align="C")


def gerar_relatorio_pdf(
    df_resultados: pd.DataFrame,
    metricas: Dict[str, Any],
    coluna_texto: str,
    max_amostras: int = 25
) -> bytes:
    if not FPDF_INSTALADO:
        raise RuntimeError("A biblioteca 'fpdf2' não está instalada no ambiente.")

    pdf = RelatorioExecutivoPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, sanitizar_texto_latin1("Relatório Executivo de Auditoria de Sentimentos"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(
        0, 5,
        sanitizar_texto_latin1(
            "Este relatório sintetiza a avaliação analítica de sentimentos de avaliações de e-commerce "
            "processadas pelo modelo de Regressão Logística TF-IDF. A heurística de negação permanece "
            "opcional e desativada por padrão."
        )
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 34, 64)
    pdf.cell(0, 8, sanitizar_texto_latin1("1. Indicadores Chave de Desempenho (KPIs)"), ln=True)
    pdf.ln(1)

    largura_col = (pdf.w - 30) / 4.0
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(210, 215, 225)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(60, 60, 60)

    pdf.cell(largura_col, 6, sanitizar_texto_latin1("Total de Avaliações"), 1, 0, "C", fill=True)
    pdf.cell(largura_col, 6, sanitizar_texto_latin1("Sentimento Positivo"), 1, 0, "C", fill=True)
    pdf.cell(largura_col, 6, sanitizar_texto_latin1("Sentimento Negativo"), 1, 0, "C", fill=True)
    pdf.cell(largura_col, 6, sanitizar_texto_latin1("Confiança Média"), 1, 1, "C", fill=True)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(largura_col, 8, str(metricas["total"]), 1, 0, "C")

    pdf.set_text_color(30, 130, 60)
    pdf.cell(largura_col, 8, f"{metricas['pct_positivos']}% ({metricas['positivos']})", 1, 0, "C")

    pdf.set_text_color(180, 40, 40)
    pdf.cell(largura_col, 8, f"{metricas['pct_negativos']}% ({metricas['negativos']})", 1, 0, "C")

    pdf.set_text_color(20, 20, 20)
    pdf.cell(largura_col, 8, f"{metricas['confianca_media']}%", 1, 1, "C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 34, 64)
    pdf.cell(0, 8, sanitizar_texto_latin1("2. Metodologia e Integridade Analítica"), ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    texto_metodologia = (
        f"- Modelo Principal: Regressão Logística com vetorização TF-IDF (1-2 n-gramas).\n"
        f"- Pré-processamento: minúsculas e sem acentos (igual ao treino). Heurística opcional e experimental.\n"
        f"- Intervenções Heurísticas: {metricas['total_heuristicas']} registros com override."
    )
    pdf.multi_cell(0, 5, sanitizar_texto_latin1(texto_metodologia))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 34, 64)
    pdf.cell(0, 8, sanitizar_texto_latin1(f"3. Amostra Analítica de Avaliações (Top {min(max_amostras, len(df_resultados))})"), ln=True)
    pdf.ln(1)

    col_w_id = 12
    col_w_sent = 28
    col_w_conf = 22
    col_w_text = (pdf.w - 30) - (col_w_id + col_w_sent + col_w_conf)

    pdf.set_fill_color(235, 240, 248)
    pdf.set_draw_color(200, 205, 215)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(20, 20, 20)

    pdf.cell(col_w_id, 7, "#", 1, 0, "C", fill=True)
    pdf.cell(col_w_sent, 7, sanitizar_texto_latin1("Sentimento"), 1, 0, "C", fill=True)
    pdf.cell(col_w_conf, 7, sanitizar_texto_latin1("Confiança"), 1, 0, "C", fill=True)
    pdf.cell(col_w_text, 7, sanitizar_texto_latin1("Trecho da Avaliação do Cliente"), 1, 1, "L", fill=True)

    pdf.set_font("Helvetica", "", 8)
    amostras = df_resultados.head(max_amostras)

    for i, (_, row) in enumerate(amostras.iterrows(), start=1):
        sentimento = str(row.get("sentimento_predito", "N/A"))
        confianca = f"{row.get('confianca_pct', 0.0):.1f}%"
        texto_original = str(row.get(coluna_texto, ""))
        texto_curto = (texto_original[:90] + "...") if len(texto_original) > 90 else texto_original

        if sentimento == "Positivo":
            pdf.set_text_color(30, 130, 60)
        else:
            pdf.set_text_color(180, 40, 40)

        pdf.cell(col_w_id, 6, str(i), 1, 0, "C")
        pdf.cell(col_w_sent, 6, sanitizar_texto_latin1(sentimento), 1, 0, "C")

        pdf.set_text_color(50, 50, 50)
        pdf.cell(col_w_conf, 6, confianca, 1, 0, "C")
        pdf.cell(col_w_text, 6, sanitizar_texto_latin1(texto_curto), 1, 1, "L")

    return bytes(pdf.output())


def historico_para_dataframe(historico: List[Dict[str, Any]]) -> pd.DataFrame:
    if not historico:
        return pd.DataFrame(columns=["texto", "sentimento_predito", "confianca_pct", "data"])
    linhas = []
    for item in historico:
        linhas.append({
            "texto": item.get("texto", ""),
            "sentimento_predito": item.get("sentimento", ""),
            "confianca_pct": round(float(item.get("confianca", 0.0)), 2),
            "data": item.get("data", ""),
            "heuristica_seguranca": bool(item.get("heuristica_acionada", False)),
        })
    return pd.DataFrame(linhas)


# ==============================================================================
# 5. GERENCIAMENTO ROBUSTO DE ESTADO (st.session_state)
# ==============================================================================
def inicializar_session_state() -> None:
    defaults: Dict[str, Any] = {
        "df_processado": None,
        "metricas_lote": None,
        "historico_individual": [],
        "coluna_selecionada": None,
        "ultimo_arquivo_nome": None,
        "pdf_bytes_cache": None,
        "pagina": PAGINAS[0],
        "usar_heuristica": False,
        "texto_avaliacao": "",
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


# ==============================================================================
# 6. COMPONENTES DE INTERFACE VISUAL (UI STREAMLIT)
# ==============================================================================
def renderizar_sidebar(modelo: Any, vetorizador: Any) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark"></div>'
            "<div><h2>UC9</h2><p>SENAC MT</p></div></div>",
            unsafe_allow_html=True,
        )

        pagina = st.radio(
            "Navegação",
            PAGINAS,
            key="pagina",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="about-mini"><strong style="color:#fff">Sobre o Projeto</strong><br>'
            "Projeto Final UC9 desenvolvido para análise de sentimentos em "
            "avaliações de produtos utilizando Regressão Logística.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="user-card"><strong>Luiz Dias Frederico Neto</strong>'
            "<span>Equipe UC9 - SENAC MT</span></div>",
            unsafe_allow_html=True,
        )

        if modelo is not None and vetorizador is not None:
            st.caption("Modelo e vetorizador carregados.")
        else:
            st.error("Falha nos artefatos de ML")

    return pagina


def renderizar_topo(pagina: str) -> None:
    col_a, col_b = st.columns([4, 1.4])
    with col_a:
        if pagina == "Análise de Sentimentos":
            renderizar_hero(
                "Análise de Sentimentos em E-commerce",
                "Classificação de sentimentos utilizando Regressão Logística",
            )
        elif pagina == "Dashboard":
            renderizar_hero("Dashboard", "Indicadores a partir das análises desta sessão e do corpus B2W")
        elif pagina == "Histórico de Análises":
            renderizar_hero("Histórico de Análises", "Registros desta sessão (não são dados fictícios)")
        elif pagina == "Relatórios":
            renderizar_hero("Relatórios", "Exportação em PDF e CSV a partir de resultados reais")
        elif pagina == "Modelos":
            renderizar_hero("Modelos", "Regressão Logística, TF-IDF e métricas oficiais")
        elif pagina == "Configurações":
            renderizar_hero("Configurações", "Opções que alteram o comportamento da análise")
        else:
            renderizar_hero("Sobre o Projeto", "Projeto Final UC9 - SENAC Mato Grosso")
    with col_b:
        st.write("")
        if FPDF_INSTALADO and (
            st.session_state.df_processado is not None or st.session_state.historico_individual
        ):
            df_exp, met, col_txt = obter_fonte_exportacao()
            if df_exp is not None and met is not None:
                try:
                    pdf_bytes = gerar_relatorio_pdf(df_exp, met, col_txt, max_amostras=25)
                    st.download_button(
                        "Exportar Relatório",
                        data=pdf_bytes,
                        file_name=f"relatorio_sentimentos_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as erro_pdf:
                    st.caption(f"PDF indisponível: {erro_pdf}")


def obter_fonte_exportacao() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], str]:
    if st.session_state.df_processado is not None and st.session_state.metricas_lote is not None:
        return (
            st.session_state.df_processado,
            st.session_state.metricas_lote,
            st.session_state.coluna_selecionada or "texto",
        )
    hist = historico_para_dataframe(st.session_state.historico_individual)
    if hist.empty:
        return None, None, "texto"
    return hist, calcular_metricas_resumo(hist), "texto"


def renderizar_pagina_analise(modelo: Any, vetorizador: Any, usar_heuristica: bool) -> None:
    with st.container(border=True):
        st.markdown("### Avaliação do Produto")
        st.caption("Digite a avaliação do cliente para análise de sentimento")

        texto_usuario = st.text_area(
            "Avaliação do Cliente:",
            height=130,
            max_chars=LIMITE_CARACTERES,
            placeholder="Ex: O produto chegou super rápido, excede todas as expectativas e a qualidade é excelente!",
            label_visibility="collapsed",
            key="campo_avaliacao",
        )
        n_chars = len(texto_usuario)
        col_btn, col_info = st.columns([1.2, 2.2])
        with col_btn:
            botao_analisar = st.button("Analisar Sentimento", type="primary", use_container_width=True)
        with col_info:
            st.caption(f"{n_chars} / {LIMITE_CARACTERES} caracteres")
            st.caption("O modelo classifica em Positivo ou Negativo (binário). Neutro existe só na rotulagem da base B2W pela nota.")

        if botao_analisar:
            if not texto_usuario.strip():
                st.warning("Por favor, digite um texto válido para que a predição possa ocorrer.")
            else:
                try:
                    with st.spinner("Processando texto com TF-IDF e Regressão Logística..."):
                        resultado = analisar_texto_individual(
                            texto=texto_usuario,
                            modelo=modelo,
                            vetorizador=vetorizador,
                            usar_heuristica=usar_heuristica,
                        )
                    st.session_state.historico_individual.insert(0, resultado)

                    classe_css = "result-pos" if resultado["sentimento"] == "Positivo" else "result-neg"
                    st.markdown(
                        f'<div class="{classe_css}"><h2>{resultado["sentimento"]}</h2>'
                        f'<p>{resultado["motivo"]}</p></div>',
                        unsafe_allow_html=True,
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Confiança",
                        "Regra de segurança" if resultado["heuristica_acionada"] else f"{resultado['confianca']:.2f}%",
                    )
                    c2.metric("Prob. positiva", f"{resultado['prob_positivo']:.2f}%")
                    c3.metric("Prob. negativa", f"{resultado['prob_negativo']:.2f}%")
                    st.progress(min(max(resultado["prob_positivo"] / 100.0, 0.0), 1.0))
                except Exception as erro:
                    st.error(f"Falha inesperada durante a análise: {erro}")

    col_esq, col_dir = st.columns([1.7, 1])
    with col_esq:
        with st.container(border=True):
            st.markdown("### Como funciona nossa análise")
            st.caption("Utilizamos Regressão Logística para classificar o sentimento em tempo real.")
            st.markdown(
                '<div class="steps">'
                '<div class="step"><div class="dot dot-g"></div><h4>1. Pré-processamento</h4>'
                "<p>O texto vai para minúsculas e sem acentos, o mesmo tratamento do treino, antes de vetorizar.</p></div>"
                '<div class="step"><div class="dot dot-p"></div><h4>2. Análise pelo Modelo</h4>'
                "<p>O vetorizador TF-IDF e a Regressão Logística geram a probabilidade das classes.</p></div>"
                '<div class="step"><div class="dot dot-v"></div><h4>3. Classificação</h4>'
                "<p>O sentimento previsto pelo modelo é Positivo ou Negativo.</p></div>"
                "</div>",
                unsafe_allow_html=True,
            )
    with col_dir:
        with st.container(border=True):
            st.markdown("### Estatísticas do Modelo")
            m = carregar_metricas_oficiais()
            if m:
                st.markdown(
                    f'<div class="stat-row"><span>Acurácia</span><span class="stat-ok">{m["acuracia"]}%</span></div>'
                    f'<div class="stat-row"><span>Precisão</span><span class="stat-ok">{m["precisao"]}%</span></div>'
                    f'<div class="stat-row"><span>Recall</span><span class="stat-ok">{m["recall"]}%</span></div>'
                    f'<div class="stat-row"><span>F1-Score</span><span class="stat-warn">{m["f1"]}%</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Conjunto de teste (n = {m['n_teste']:,}). {m['atualizado']}.")
            else:
                st.info("Métricas ainda não encontradas em models/metricas.json.")


def renderizar_pagina_dashboard() -> None:
    hist = historico_para_dataframe(st.session_state.historico_individual)
    st.markdown("#### Análises desta sessão")
    if hist.empty and st.session_state.df_processado is None:
        st.info("Ainda não há análises nesta sessão. Use a página Análise de Sentimentos ou envie um CSV em Relatórios.")
    else:
        if st.session_state.metricas_lote is not None:
            met = st.session_state.metricas_lote
            fonte = f"Lote: {st.session_state.ultimo_arquivo_nome or 'CSV'}"
        else:
            met = calcular_metricas_resumo(hist)
            fonte = "Histórico individual da sessão"
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", f"{met['total']:,}")
        k2.metric("Positivas", f"{met['positivos']:,}", f"{met['pct_positivos']}%")
        k3.metric("Negativas", f"{met['negativos']:,}", f"{met['pct_negativos']}%")
        k4.metric("Confiança média", f"{met['confianca_media']}%")
        st.caption(fonte)
        df_grafico = pd.DataFrame({
            "Sentimento": ["Positivo", "Negativo"],
            "Quantidade": [met["positivos"], met["negativos"]],
        }).set_index("Sentimento")
        st.bar_chart(df_grafico)
        if not hist.empty:
            st.markdown("#### Histórico resumido")
            st.dataframe(hist.head(20), use_container_width=True, hide_index=True)

    resumo = carregar_resumo_b2w()
    st.markdown("#### Corpus B2W-Reviews01 (rótulo pela nota, não é previsão do modelo)")
    if not resumo:
        st.info("Resumo B2W não encontrado. Execute `python scripts/preparar_b2w.py` após baixar a base.")
        return
    d = resumo["distribuicao_preparado"]
    p = resumo["percentual_preparado"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Avaliações preparadas", f"{resumo['n_preparado']:,}")
    b2.metric("Positivas (nota 4-5)", f"{d.get('Positivo', 0):,}", f"{p.get('Positivo', 0)}%")
    b3.metric("Neutras (nota 3)", f"{d.get('Neutro', 0):,}", f"{p.get('Neutro', 0)}%")
    b4.metric("Negativas (nota 1-2)", f"{d.get('Negativo', 0):,}", f"{p.get('Negativo', 0)}%")
    st.caption(
        f"Origem: {resumo['origem']}. Licença {resumo['licenca']} (crédito: {resumo['credito']}). "
        f"Original {resumo['n_original']:,} linhas; removidas {resumo['n_removidos_sem_texto_ou_nota']:,} sem texto."
    )
    df_b2w = pd.DataFrame({
        "Sentimento (nota)": ["Positivo", "Neutro", "Negativo"],
        "Quantidade": [d.get("Positivo", 0), d.get("Neutro", 0), d.get("Negativo", 0)],
    }).set_index("Sentimento (nota)")
    st.bar_chart(df_b2w)
    amostra = carregar_amostra_b2w()
    if amostra is not None:
        st.caption("Amostra de 80 linhas (data/b2w_amostra.csv). A base completa não é carregada na interface.")
        st.dataframe(
            amostra[["product_name", "overall_rating", "sentimento_rotulo", "review_text"]].head(15),
            use_container_width=True,
            hide_index=True,
        )


def renderizar_pagina_historico() -> None:
    hist = historico_para_dataframe(st.session_state.historico_individual)
    if hist.empty:
        st.info("Nenhuma análise individual nesta sessão.")
        return
    busca = st.text_input("Pesquisar no texto")
    filtro = st.radio("Filtro", ["Todos", "Positivo", "Negativo"], horizontal=True)
    df = hist
    if busca.strip():
        df = df[df["texto"].str.contains(busca, case=False, na=False)]
    if filtro != "Todos":
        df = df[df["sentimento_predito"] == filtro]
    ordem = st.selectbox("Ordenar por", ["Mais recentes", "Confiança (maior)", "Confiança (menor)"])
    if ordem == "Confiança (maior)":
        df = df.sort_values("confianca_pct", ascending=False)
    elif ordem == "Confiança (menor)":
        df = df.sort_values("confianca_pct", ascending=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("Limpar histórico desta sessão"):
        st.session_state.historico_individual = []
        st.rerun()


def renderizar_pagina_relatorios(modelo: Any, vetorizador: Any, usar_heuristica: bool) -> None:
    st.markdown("#### Análise em lote via CSV")
    st.write("Envie um arquivo com ao menos uma coluna de texto. O modelo existente será aplicado, sem retreino.")
    arquivo_enviado = st.file_uploader("Carregue um arquivo .CSV:", type=["csv"])

    if arquivo_enviado is not None:
        try:
            df_upload = pd.read_csv(arquivo_enviado, on_bad_lines="skip")
            if df_upload.empty:
                st.warning("O arquivo CSV carregado está vazio.")
            else:
                st.write(f"Arquivo carregado: **{arquivo_enviado.name}** ({len(df_upload):,} linhas)")
                colunas_disponiveis = df_upload.columns.tolist()
                colunas_candidatas = [
                    c for c in colunas_disponiveis
                    if any(k in c.lower() for k in ["text", "review", "coment", "aval", "texto"])
                ]
                indice_padrao = colunas_disponiveis.index(colunas_candidatas[0]) if colunas_candidatas else 0
                col_selecionada = st.selectbox(
                    "Selecione a coluna que contém o texto da avaliação:",
                    options=colunas_disponiveis,
                    index=indice_padrao,
                )
                if st.button("Processar Base Completa", type="primary"):
                    with st.spinner("Executando vetorização TF-IDF e classificação em lote..."):
                        df_resultado = analisar_dataframe_lote(
                            df=df_upload,
                            coluna_texto=col_selecionada,
                            modelo=modelo,
                            vetorizador=vetorizador,
                            usar_heuristica=usar_heuristica,
                        )
                        metricas = calcular_metricas_resumo(df_resultado)
                        st.session_state.df_processado = df_resultado
                        st.session_state.metricas_lote = metricas
                        st.session_state.coluna_selecionada = col_selecionada
                        st.session_state.ultimo_arquivo_nome = arquivo_enviado.name
                        st.session_state.pdf_bytes_cache = None
                    st.success("Processamento em lote concluído.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo CSV: {e}")

    df_exp, met, col_txt = obter_fonte_exportacao()
    if df_exp is None or met is None:
        st.info("Gere análises individuais ou processe um CSV para habilitar a exportação.")
        return

    st.markdown("#### Resultados disponíveis para exportação")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total", f"{met['total']:,}")
    k2.metric("Positivas", f"{met['positivos']:,}")
    k3.metric("Negativas", f"{met['negativos']:,}")
    k4.metric("Confiança média", f"{met['confianca_media']}%")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if not FPDF_INSTALADO:
            st.warning("Instale o pacote fpdf2 para habilitar relatórios em PDF.")
        else:
            try:
                if st.session_state.pdf_bytes_cache is None:
                    st.session_state.pdf_bytes_cache = gerar_relatorio_pdf(
                        df_resultados=df_exp,
                        metricas=met,
                        coluna_texto=col_txt,
                        max_amostras=25,
                    )
                st.download_button(
                    label="Baixar Relatório Executivo (.pdf)",
                    data=st.session_state.pdf_bytes_cache,
                    file_name=f"relatorio_executivo_sentimentos_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as erro_pdf:
                st.error(f"Erro ao gerar o arquivo PDF: {erro_pdf}")
    with col_exp2:
        csv_dados = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar Base Processada (.csv)",
            data=csv_dados,
            file_name=f"avaliacoes_classificadas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def renderizar_pagina_modelos() -> None:
    st.markdown("""
    O classificador em uso é o artefato já treinado, sem retreino nesta interface.

    1. **Pré-processamento** (`preprocessamento.py`): minúsculas e remoção de acentos, o mesmo tratamento da coluna `review_text_processed` do treino.
    2. **Vetorizador TF-IDF**: até 5.000 termos, unigramas e bigramas (`models/vetorizador_tfidf.pkl`).
    3. **Regressão Logística**: classificação binária positivo/negativo (`models/modelo_regressao_logistica.pkl`).
    4. **Heurística de palavras negativas**: experimental e **desativada por padrão** (reduziu a acurácia de 94,57% para 77,08% no teste).
    """)
    m = carregar_metricas_oficiais()
    if not m:
        st.info("Métricas ainda não geradas. Execute `python scripts/treinar_modelo.py`.")
        return
    st.table(pd.DataFrame({
        "Métrica": ["Acurácia", "Precisão (macro)", "Recall (macro)", "F1-Score (macro)", "Baseline (classe majoritária)"],
        "Valor (%)": [m["acuracia"], m["precisao"], m["recall"], m["f1"], m["baseline_acuracia"]],
    }))
    st.caption(
        f"Calculadas no conjunto de teste (20%, n = {m['n_teste']:,}, estratificado). "
        f"Treino n = {m['n_treino']:,}. Gerado em {m['atualizado']}."
    )
    st.write("Arquivos utilizados: `models/modelo_regressao_logistica.pkl`, `models/vetorizador_tfidf.pkl`, `models/metricas.json`.")


def renderizar_pagina_configuracoes() -> bool:
    st.write("Somente opções com efeito real na análise.")
    usar = st.checkbox(
        "Ativar Heurística de Negação",
        key="usar_heuristica",
        help="Experimental. No conjunto de teste ela reduziu a acurácia de 94,57% para 77,08%, pois termos como caro ou defeito também aparecem em avaliações positivas.",
    )
    if usar:
        st.warning("Heurística ativada. O resultado pode divergir da predição pura do modelo.")
    else:
        st.caption("Heurística desativada (recomendado). A classificação sai só da Regressão Logística.")
    st.caption("Tema escuro definido em .streamlit/config.toml e tema.py.")
    return usar


def renderizar_pagina_sobre() -> None:
    resumo = carregar_resumo_b2w()
    st.markdown("""
    **Objetivo acadêmico.** Projeto Final da UC9 do SENAC Mato Grosso: classificar o sentimento de avaliações de e-commerce com Machine Learning.

    **Tecnologias.** Python, Streamlit, scikit-learn (TF-IDF + Regressão Logística), pandas, joblib, fpdf2.

    **Equipe**
    - Luiz Dias Frederico Neto
    - Bruno Kenzo Nobre Miyagawa
    - João Lucas Fernandes de Barros

    **Instituição.** SENAC Mato Grosso.

    **Limitações.** O modelo salvo é binário (não prevê Neutro). Não detecta ironia. Treinado em avaliações de e-commerce brasileiro.
    """)
    if resumo:
        st.markdown("**Base B2W-Reviews01**")
        st.write(
            f"Origem: {resumo['origem']}. Repositório: {resumo['repositorio_oficial']}. "
            f"Licença {resumo['licenca']}, crédito a {resumo['credito']}."
        )
        st.write(
            "Rotulagem de três classes usada só no arquivo preparado, pela nota: "
            "1-2 Negativo, 3 Neutro, 4-5 Positivo. Isso não altera o modelo treinado."
        )


def renderizar_rodape() -> None:
    st.markdown(
        '<div class="footer-note">Projeto Final UC9 - SENAC Mato Grosso<br>'
        "Desenvolvido por Luiz Dias Frederico Neto, Bruno Kenzo Nobre Miyagawa e João Lucas Fernandes de Barros</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# 7. FUNÇÃO PRINCIPAL DE CONTROLE (PONTO DE ENTRADA)
# ==============================================================================
def main() -> None:
    inicializar_session_state()

    modelo, vetorizador = None, None
    try:
        modelo, vetorizador = carregar_artefatos(CAMINHO_MODELO, CAMINHO_VETORIZADOR)
    except FileNotFoundError as err:
        st.error(f"**Erro de Artefato Ausente:** {err}")
        st.info("Certifique-se de que os arquivos `modelo_regressao_logistica.pkl` e `vetorizador_tfidf.pkl` estejam em models/.")
        st.stop()
    except Exception as err:
        st.error(f"**Erro Crítico no Carregamento:** {err}")
        st.stop()

    pagina = renderizar_sidebar(modelo, vetorizador)
    usar_heuristica = bool(st.session_state.usar_heuristica)
    renderizar_topo(pagina)

    if pagina == "Análise de Sentimentos":
        renderizar_pagina_analise(modelo, vetorizador, usar_heuristica)
    elif pagina == "Dashboard":
        renderizar_pagina_dashboard()
    elif pagina == "Histórico de Análises":
        renderizar_pagina_historico()
    elif pagina == "Relatórios":
        renderizar_pagina_relatorios(modelo, vetorizador, usar_heuristica)
    elif pagina == "Modelos":
        renderizar_pagina_modelos()
    elif pagina == "Configurações":
        renderizar_pagina_configuracoes()
    else:
        renderizar_pagina_sobre()

    renderizar_rodape()


if __name__ == "__main__":
    main()
