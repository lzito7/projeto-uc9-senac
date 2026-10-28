"""
================================================================================
PROJETO FINAL UC9 - ANÁLISE DE SENTIMENTOS EM E-COMMERCE (SENAC-MT)
================================================================================
Equipe Acadêmica:
- Luiz Dias Frederico Neto
- Bruno Kenzo Nobre Miyagawa
- João Lucas Fernandes de Barros

Arquitetura de Software Refatorada (Padrão Produção / Clean Architecture):
1. Configurações Globais, Constantes e Compilação de Regex
2. Camada de ML & NLP (Carregamento Otimizado com @st.cache_resource)
3. Camada de Lógica de Negócio (Inferência Individual e Vetorizada em Lote)
4. Camada de Emissão de Relatórios Executivos em PDF (fpdf2 com suporte UTF-8/Latin-1)
5. Gerenciamento Resiliente de Estado (st.session_state anti-NameError)
6. Camada de Apresentação Modular (Streamlit Wide Layout, KPIs e Abas)
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

# Importação condicional e defensiva do fpdf2
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

# Lista canônica de termos críticos de insatisfação / negação explícita
TERMOS_NEGATIVOS = [
    "péssimo", "pessimo", "ruim", "horrível", "horrivel",
    "decepcionou", "estragado", "defeituoso", "caro", "arrependi",
    "demorou", "fraude", "péssima", "pessima", "defeito", "pior",
    "estragou", "danificado", "falha", "lento", "descartável",
    "não gostei", "nao gostei", "não recomendo", "nao recomendo", "não funciona", "nao funciona",
    "não chegou", "nao chegou", "não vale", "nao vale"
]

# Regex compilada com limites de palavra (\b) para prevenir falsos positivos em substrings
PADRAO_NEGACAO_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(termo) for termo in TERMOS_NEGATIVOS) + r")\b",
    re.IGNORECASE
)

# Configuração da página Streamlit (Layout wide para melhor dashboard e relatórios)
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
    """
    Carrega o modelo preditivo e o vetorizador TF-IDF do disco com cache de recurso.
    Realiza checagem defensiva de existência dos arquivos e métodos necessários.
    """
    if not caminho_mod.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: '{caminho_mod.name}'.")
    if not caminho_vet.exists():
        raise FileNotFoundError(f"Arquivo do vetorizador não encontrado: '{caminho_vet.name}'.")

    modelo = joblib.load(caminho_mod)
    vetorizador = joblib.load(caminho_vet)

    # Validação estrutural dos artefatos carregados
    if not hasattr(modelo, "predict") or not hasattr(modelo, "predict_proba"):
        raise TypeError("O objeto do modelo carregado não possui os métodos 'predict' e 'predict_proba'.")
    if not hasattr(vetorizador, "transform"):
        raise TypeError("O vetorizador carregado não possui o método 'transform'.")

    return modelo, vetorizador


# ==============================================================================
# 3. CAMADA DE LÓGICA DE NEGÓCIO E INFERÊNCIA NLP
# ==============================================================================
def verificar_heuristica_insatisfacao(texto: str) -> Tuple[bool, List[str]]:
    """
    Avalia a presença de termos críticos de negação/insatisfação usando regex com limites de palavra.
    Retorna se há gatilho e a lista de termos únicos detectados.
    """
    if not texto:
        return False, []
    encontrados = PADRAO_NEGACAO_REGEX.findall(texto.lower())
    termos_unicos = list(dict.fromkeys(encontrados))
    return len(termos_unicos) > 0, termos_unicos


def analisar_texto_individual(
    texto: str,
    modelo: Any,
    vetorizador: Any,
    usar_heuristica: bool = True
) -> Dict[str, Any]:
    """
    Executa a inferência de sentimento em uma avaliação individual.
    Combina a previsão estatística do modelo com a heurística de segurança, se ativada.
    """
    texto_limpo = texto.strip()
    if not texto_limpo:
        raise ValueError("O texto fornecido está vazio.")

    # 1. Avaliação heurística
    heuristica_acionada, termos_encontrados = (
        verificar_heuristica_insatisfacao(texto_limpo) if usar_heuristica else (False, [])
    )

    # 2. Vetorização e Inferência ML
    vetor = vetorizador.transform([normalizar_texto(texto_limpo)])
    predicao_ml = int(modelo.predict(vetor)[0])
    probabilidades = modelo.predict_proba(vetor)[0]

    prob_negativo = float(probabilidades[0])
    prob_positivo = float(probabilidades[1]) if len(probabilidades) > 1 else (1.0 - prob_negativo)

    # 3. Decisão final (Heurística de segurança ou predição pura do modelo)
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
        "motivo": motivo
    }


def analisar_dataframe_lote(
    df: pd.DataFrame,
    coluna_texto: str,
    modelo: Any,
    vetorizador: Any,
    usar_heuristica: bool = True
) -> pd.DataFrame:
    """
    Executa a inferência vetorizada em lote sobre um DataFrame, otimizando o uso de memória e CPU.
    """
    if coluna_texto not in df.columns:
        raise KeyError(f"A coluna '{coluna_texto}' não foi encontrada no DataFrame.")

    # Cria cópia de trabalho e trata valores nulos/vazios
    df_proc = df.copy()
    series_texto = df_proc[coluna_texto].fillna("").astype(str).str.strip()

    # Vetorização em lote (alta performance com chamada única a transform)
    X_tfidf = vetorizador.transform(series_texto.map(normalizar_texto))
    predicoes_ml = modelo.predict(X_tfidf)
    probabilidades = modelo.predict_proba(X_tfidf)

    prob_neg = probabilidades[:, 0]
    prob_pos = probabilidades[:, 1] if probabilidades.shape[1] > 1 else (1.0 - prob_neg)
    confiancas = probabilidades.max(axis=1) * 100.0

    sentimentos = ["Positivo" if p == 1 else "Negativo" for p in predicoes_ml]
    heuristica_flags = [False] * len(df_proc)
    termos_lista = [""] * len(df_proc)

    # Aplicação de heurística se ativada
    if usar_heuristica:
        for idx, txt in enumerate(series_texto):
            acionou, termos = verificar_heuristica_insatisfacao(txt)
            if acionou:
                heuristica_flags[idx] = True
                termos_lista[idx] = ", ".join(termos)
                sentimentos[idx] = "Negativo"
                confiancas[idx] = prob_neg[idx] * 100.0

    # Enriquecimento do DataFrame
    df_proc["sentimento_predito"] = sentimentos
    df_proc["confianca_pct"] = confiancas.round(2)
    df_proc["prob_positivo_pct"] = (prob_pos * 100.0).round(2)
    df_proc["prob_negativo_pct"] = (prob_neg * 100.0).round(2)
    df_proc["heuristica_seguranca"] = heuristica_flags
    df_proc["termos_heuristica"] = termos_lista

    return df_proc


def calcular_metricas_resumo(df_resultados: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula os KPIs executivos consolidados a partir do DataFrame de resultados processados.
    """
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
    """
    Higieniza o texto para compatibilidade estrita com fontes padrão Latin-1 do FPDF.
    Normaliza caracteres tipográficos (aspas curvas, travessões) e substitui símbolos incompatíveis.
    """
    if texto is None:
        return ""
    texto_str = str(texto)

    # Substituição de caracteres tipográficos comuns que quebram FPDF
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

    # Codifica e decodifica em latin-1 substituindo caracteres não suportados por '?'
    # Isso preserva integralmente acentos em português (á, é, í, ó, ú, â, ê, õ, ã, ç, etc.)
    return texto_str.encode("latin-1", "replace").decode("latin-1")


if FPDF_INSTALADO:
    class RelatorioExecutivoPDF(FPDF):
        """
        Classe customizada para emissão de relatórios executivos acadêmicos e empresariais.
        Garante layout padronizado, cabeçalho institucional, paginação e margens de 15mm.
        """
        def header(self):
            # Banner Superior Institucional SENAC
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(15, 34, 64)  # Azul Marinho SENAC
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
    """
    Constrói o documento PDF executivo utilizando fpdf2 e retorna os bytes brutos para download.
    """
    if not FPDF_INSTALADO:
        raise RuntimeError("A biblioteca 'fpdf2' não está instalada no ambiente.")

    pdf = RelatorioExecutivoPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título do Relatório
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, sanitizar_texto_latin1("Relatório Executivo de Auditoria de Sentimentos"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(
        0, 5,
        sanitizar_texto_latin1(
            "Este relatório sintetiza a avaliação analítica de sentimentos de avaliações de e-commerce "
            "processadas pelo modelo de Regressão Logística TF-IDF com suporte a heurística de negação."
        )
    )
    pdf.ln(4)

    # ---------------------------------------------------------
    # Bloco de KPIs / Resumo Executivo
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 34, 64)
    pdf.cell(0, 8, sanitizar_texto_latin1("1. Indicadores Chave de Desempenho (KPIs)"), ln=True)
    pdf.ln(1)

    # Grid de KPIs executivo
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
    
    pdf.set_text_color(30, 130, 60)  # Verde
    pdf.cell(largura_col, 8, f"{metricas['pct_positivos']}% ({metricas['positivos']})", 1, 0, "C")
    
    pdf.set_text_color(180, 40, 40)   # Vermelho
    pdf.cell(largura_col, 8, f"{metricas['pct_negativos']}% ({metricas['negativos']})", 1, 0, "C")
    
    pdf.set_text_color(20, 20, 20)
    pdf.cell(largura_col, 8, f"{metricas['confianca_media']}%", 1, 1, "C")
    pdf.ln(5)

    # ---------------------------------------------------------
    # Bloco Metodológico e Integridade dos Dados
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 34, 64)
    pdf.cell(0, 8, sanitizar_texto_latin1("2. Metodologia e Integridade Analítica"), ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    texto_metodologia = (
        f"- Modelo Principal: Regressão Logística com vetorização TF-IDF (1-2 n-gramas).\n"
        f"- Pré-processamento: minúsculas e sem acentos (igual ao treino). Heurística opcional e experimental.\n"
        f"- Intervenções Heurísticas: {metricas['total_heuristicas']} registros com override crítico para proteção da marca."
    )
    pdf.multi_cell(0, 5, sanitizar_texto_latin1(texto_metodologia))
    pdf.ln(4)

    # ---------------------------------------------------------
    # Amostra de Registros Auditados
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 34, 64)
    pdf.cell(0, 8, sanitizar_texto_latin1(f"3. Amostra Analítica de Avaliações (Top {min(max_amostras, len(df_resultados))})"), ln=True)
    pdf.ln(1)

    # Larguras das colunas calculadas para preencher exatamente 180mm úteis do A4
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


# ==============================================================================
# 5. GERENCIAMENTO ROBUSTO DE ESTADO (st.session_state)
# ==============================================================================
def inicializar_session_state() -> None:
    """
    Garante a presença e tipagem de todas as variáveis de estado de sessão,
    evitando quebras de NameError ou KeyError em reexecuções fora de ordem.
    """
    defaults: Dict[str, Any] = {
        "df_processado": None,
        "metricas_lote": None,
        "historico_individual": [],
        "coluna_selecionada": None,
        "ultimo_arquivo_nome": None,
        "pdf_bytes_cache": None
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


# ==============================================================================
# 6. COMPONENTES DE INTERFACE VISUAL (UI STREAMLIT)
# ==============================================================================
def renderizar_sidebar(modelo: Any, vetorizador: Any) -> Tuple[bool, bool]:
    """
    Renderiza os painéis de navegação, status dos modelos e configurações na barra lateral.
    """
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/86/Senac_logo.svg", width=140)
        st.markdown("### Projeto Final UC9")
        st.markdown("**Análise de Sentimentos em E-commerce**")
        st.caption("SENAC Mato Grosso - Inteligência Artificial")
        
        st.divider()
        st.markdown("#### Parâmetros de Inferência")
        usar_heuristica = st.checkbox(
            "Ativar Heurística de Negação",
            value=False,
            help="Experimental. No conjunto de teste ela reduziu a acurácia de 94,57% para 77,08%, pois termos como caro ou defeito também aparecem em avaliações positivas."
        )

        st.divider()
        st.markdown("#### Status dos Artefatos (.pkl)")
        if modelo is not None and vetorizador is not None:
            st.success("Regressão Logística Carregada")
            st.success("Vetorizador TF-IDF Carregado")
            st.caption(f"Tamanho do vocabulário TF-IDF: {len(getattr(vetorizador, 'vocabulary_', {})):,} termos")
        else:
            st.error("Falha nos Artefatos de ML")

        st.divider()
        st.markdown("#### Integrantes da Equipe")
        st.markdown("""
        - **Luiz Dias Frederico Neto**
        - **Bruno Kenzo Nobre Miyagawa**
        - **João Lucas Fernandes de Barros**
        """)
        
        st.caption("Versão de Produção 2.1 | Refatorada")
        
    return usar_heuristica, FPDF_INSTALADO


def renderizar_aba_individual(modelo: Any, vetorizador: Any, usar_heuristica: bool) -> None:
    """
    Renderiza a interface para análise em tempo real de uma única avaliação.
    """
    st.subheader("Classificação Individual em Tempo Real")
    st.write("Insira uma avaliação de cliente abaixo para inspecionar a decisão do algoritmo.")

    exemplo_padrao = "O produto chegou no prazo, muito bem embalado e superou todas as expectativas!"
    
    with st.form("form_analise_individual"):
        texto_usuario = st.text_area(
            "Avaliação do Cliente:",
            value=exemplo_padrao,
            height=110,
            placeholder="Digite ou cole aqui o texto da avaliação..."
        )
        botao_analisar = st.form_submit_button("Analisar Sentimento", type="primary", use_container_width=True)

    if botao_analisar:
        if not texto_usuario.strip():
            st.warning("Por favor, digite um texto válido para que a predição possa ocorrer.")
            return

        try:
            with st.spinner("Processando texto com TF-IDF e Regressão Logística..."):
                resultado = analisar_texto_individual(
                    texto=texto_usuario,
                    modelo=modelo,
                    vetorizador=vetorizador,
                    usar_heuristica=usar_heuristica
                )

            # Armazena no histórico da sessão
            st.session_state.historico_individual.insert(0, resultado)

            st.divider()
            st.markdown("### Resultado da Inferência")

            col1, col2, col3 = st.columns([1.5, 1, 1.5])
            with col1:
                if resultado["sentimento"] == "Positivo":
                    st.success("### Sentimento Positivo")
                else:
                    st.error("### Sentimento Negativo")
            
            with col2:
                st.metric("Confiança Estimada", "Regra de segurança" if resultado["heuristica_acionada"] else f"{resultado['confianca']:.2f}%")
            
            with col3:
                st.info(f"**Nota Técnica:**\n{resultado['motivo']}")

            # Barras de Probabilidade Comparativa
            st.write("**Probabilidades do modelo" + (" (antes da regra de segurança):**" if resultado["heuristica_acionada"] else ":**"))
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                st.caption(f"Probabilidade Positiva: {resultado['prob_positivo']:.2f}%")
                st.progress(resultado['prob_positivo'] / 100.0)
            with col_bar2:
                st.caption(f"Probabilidade Negativa: {resultado['prob_negativo']:.2f}%")
                st.progress(resultado['prob_negativo'] / 100.0)

            if resultado["sentimento"] == "Positivo":
                pass

        except Exception as erro:
            st.error(f"Falha inesperada durante a análise: {erro}")


def renderizar_aba_lote(modelo: Any, vetorizador: Any, usar_heuristica: bool) -> None:
    """
    Renderiza a interface de processamento em massa via upload de CSV,
    exibição de KPIs executivos, filtros interativos e exportação em PDF e CSV.
    """
    st.subheader("Análise em Lote via Arquivo CSV")
    st.write("Faça upload de uma planilha de avaliações para processamento vetorizado em alta performance.")

    arquivo_enviado = st.file_uploader(
        "Carregue um arquivo .CSV:",
        type=["csv"],
        help="O arquivo deve conter ao menos uma coluna com o texto das avaliações."
    )

    if arquivo_enviado is not None:
        try:
            # Carregamento resiliente do CSV
            df_upload = pd.read_csv(arquivo_enviado, on_bad_lines="skip")
            
            if df_upload.empty:
                st.warning("O arquivo CSV carregado está vazio.")
                return

            st.write(f"Arquivo carregado: **{arquivo_enviado.name}** ({len(df_upload):,} linhas detectadas)")

            # Identificação inteligente ou seleção manual da coluna de texto
            colunas_disponiveis = df_upload.columns.tolist()
            colunas_candidatas = [c for c in colunas_disponiveis if any(k in c.lower() for k in ["text", "review", "coment", "aval", "texto"])]
            indice_padrao = colunas_disponiveis.index(colunas_candidatas[0]) if colunas_candidatas else 0

            col_selecionada = st.selectbox(
                "Selecione a coluna que contém o texto da avaliação:",
                options=colunas_disponiveis,
                index=indice_padrao
            )

            col_btn1, _ = st.columns([1, 3])
            with col_btn1:
                iniciar_proc = st.button("Processar Base Completa", type="primary", use_container_width=True)

            if iniciar_proc:
                with st.spinner("Executando vetorização TF-IDF e classificação em lote..."):
                    df_resultado = analisar_dataframe_lote(
                        df=df_upload,
                        coluna_texto=col_selecionada,
                        modelo=modelo,
                        vetorizador=vetorizador,
                        usar_heuristica=usar_heuristica
                    )
                    metricas = calcular_metricas_resumo(df_resultado)

                    # Persistência segura em session_state para evitar perda de dados em re-render
                    st.session_state.df_processado = df_resultado
                    st.session_state.metricas_lote = metricas
                    st.session_state.coluna_selecionada = col_selecionada
                    st.session_state.ultimo_arquivo_nome = arquivo_enviado.name
                    st.session_state.pdf_bytes_cache = None  # Invalida cache de PDF antigo

                st.success("Processamento em lote concluído com sucesso!")

        except Exception as e:
            st.error(f"Erro ao processar o arquivo CSV: {e}")
            return

    # Validação rigorosa de estado antes de renderizar dashboard e exportação
    if st.session_state.df_processado is not None and st.session_state.metricas_lote is not None:
        df_resultados = st.session_state.df_processado
        metricas = st.session_state.metricas_lote
        coluna_texto = st.session_state.coluna_selecionada or "review_text"

        st.divider()
        st.markdown("### Dashboard Executivo de Resultados")

        # Cards de KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total de Avaliações", f"{metricas['total']:,}")
        kpi2.metric("Positivas", f"{metricas['positivos']:,}", f"{metricas['pct_positivos']}%", delta_color="normal")
        kpi3.metric("Negativas", f"{metricas['negativos']:,}", f"-{metricas['pct_negativos']}%", delta_color="inverse")
        kpi4.metric("Confiança Média", f"{metricas['confianca_media']}%")

        # Gráfico Comparativo Nativo
        st.write("#### Distribuição de Sentimentos:")
        df_grafico = pd.DataFrame({
            "Sentimento": ["Positivo", "Negativo"],
            "Quantidade": [metricas["positivos"], metricas["negativos"]]
        }).set_index("Sentimento")
        st.bar_chart(df_grafico, color=["#1f77b4"])

        # Tabela Filtrada Interativa
        st.write("#### Visualização e Filtro dos Dados Classificados:")
        filtro_sentimento = st.radio(
            "Filtrar por Sentimento:",
            ["Todos", "Positivo", "Negativo"],
            horizontal=True
        )

        df_filtrado = df_resultados
        if filtro_sentimento != "Todos":
            df_filtrado = df_resultados[df_resultados["sentimento_predito"] == filtro_sentimento]

        colunas_exibicao = [coluna_texto, "sentimento_predito", "confianca_pct"]
        if "heuristica_seguranca" in df_filtrado.columns:
            colunas_exibicao.append("heuristica_seguranca")

        st.dataframe(
            df_filtrado[colunas_exibicao].head(100),
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------------------------------
        # Seção de Exportação Executiva (PDF e CSV)
        # ------------------------------------------------------------------
        st.divider()
        st.markdown("### Exportação de Resultados")
        
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown("#### Relatório Executivo Formal")
            if not FPDF_INSTALADO:
                st.warning("Instale o pacote `fpdf2` (`pip install fpdf2`) para habilitar relatórios em PDF.")
            else:
                try:
                    if st.session_state.pdf_bytes_cache is None:
                        with st.spinner("Gerando PDF executivo..."):
                            st.session_state.pdf_bytes_cache = gerar_relatorio_pdf(
                                df_resultados=df_resultados,
                                metricas=metricas,
                                coluna_texto=coluna_texto,
                                max_amostras=25
                            )

                    st.download_button(
                        label="Baixar Relatório Executivo (.pdf)",
                        data=st.session_state.pdf_bytes_cache,
                        file_name=f"relatorio_executivo_sentimentos_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as erro_pdf:
                    st.error(f"Erro ao gerar o arquivo PDF: {erro_pdf}")

        with col_exp2:
            st.markdown("#### Base Completa Enriquecida")
            csv_dados = df_resultados.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar Base Processada (.csv)",
                data=csv_dados,
                file_name=f"avaliacoes_classificadas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )


def renderizar_aba_documentacao() -> None:
    """
    Renderiza informações técnicas sobre a arquitetura do modelo e o projeto SENAC.
    """
    st.subheader("Metodologia e Arquitetura do Sistema")
    st.markdown("""
    ### Pipeline de Inteligência Artificial
    O projeto implementa uma solução para mineração de texto e classificação de polaridade:
    
    1. **Extração de Características (Vetorizador TF-IDF):**
       - Intervalo de n-gramas: `(1, 2)` (unigramas e bigramas para captura de expressões como *"não gostei"*).
       - Limite de vocabulário: Top 5.000 termos ponderados por relevância e frequência inversa.
       - Pré-processamento: o texto é convertido para minúsculas e sem acentos antes da vetorização, o mesmo tratamento usado no treino.
    
    2. **Classificador (Regressão Logística):**
       - Treinado com validação estratificada (80% treino / 20% teste).
       - Comparada com Naive Bayes e SVM Linear no benchmark do `deploy.ia.py` (métricas reais em `metricas.json`).
    
    3. **Heurística opcional (experimental):**
       - Interceptação com expressões regulares pré-compiladas com limites de palavra (`\\b`) para identificação de termos explícitos de insatisfação do consumidor.
       - Desativada por padrão: no conjunto de teste ela reduziu a acurácia de 94,57% para 77,08%, porque termos como "caro" ou "defeito" também aparecem em avaliações positivas. Não detecta ironia.
    
    4. **Exportação Executiva em PDF:**
       - Utilização da biblioteca `fpdf2` com tratamento rigoroso de caracteres acentuados da língua portuguesa (Latin-1 / UTF-8 safe).
       - Geração sob demanda sem estouro de memória e com paginação automática.
    """)
    arq_metricas = BASE_DIR / "models" / "metricas.json"
    st.markdown("#### Métricas do modelo")
    if arq_metricas.exists():
        m = json.loads(arq_metricas.read_text(encoding="utf-8"))
        st.table(pd.DataFrame({
            "Métrica": ["Acurácia", "Precisão (macro)", "Recall (macro)", "F1-Score (macro)", "Baseline (classe majoritária)"],
            "Valor (%)": [m["acuracia"], m["precisao"], m["recall"], m["f1"], m["baseline_acuracia"]],
        }))
        st.caption(f"Calculadas no conjunto de teste (20%, estratificado). Gerado em {m['atualizado']}.")
    else:
        st.info("Métricas ainda não geradas. Execute `python scripts/treinar_modelo.py`.")


# ==============================================================================
# 7. FUNÇÃO PRINCIPAL DE CONTROLE (PONTO DE ENTRADA)
# ==============================================================================
def main() -> None:
    """
    Ponto de entrada orquestrador da aplicação Streamlit.
    """
    # 1. Inicializa o estado de sessão de forma segura
    inicializar_session_state()

    # 2. Carrega os artefatos de ML com tratamento defensivo
    modelo, vetorizador = None, None
    try:
        modelo, vetorizador = carregar_artefatos(CAMINHO_MODELO, CAMINHO_VETORIZADOR)
    except FileNotFoundError as err:
        st.error(f"**Erro de Artefato Ausente:** {err}")
        st.info("Certifique-se de que os arquivos `modelo_regressao_logistica.pkl` e `vetorizador_tfidf.pkl` estejam na raiz do projeto.")
        st.stop()
    except Exception as err:
        st.error(f"**Erro Crítico no Carregamento:** {err}")
        st.stop()

    # 3. Renderiza a barra lateral
    usar_heuristica, fpdf_ativo = renderizar_sidebar(modelo, vetorizador)

    # 4. Título e cabeçalho principal
    renderizar_hero("Análise de Sentimentos em E-commerce", "Sistema Inteligente de Auditoria de Opinião do Cliente - Faculdade SENAC Mato Grosso")
    pass

    # 5. Navegação por Abas Modulares
    aba1, aba2, aba3 = st.tabs([
        "Análise Individual",
        "Análise em Lote & Relatório PDF",
        "ℹ️ Documentação do Modelo"
    ])

    with aba1:
        renderizar_aba_individual(modelo, vetorizador, usar_heuristica)

    with aba2:
        renderizar_aba_lote(modelo, vetorizador, usar_heuristica)

    with aba3:
        renderizar_aba_documentacao()


if __name__ == "__main__":
    main()
