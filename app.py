import streamlit as st
import joblib
from pathlib import Path

# Configuração da página
st.set_page_config(page_title="Análise de Sentimentos - SENAC", page_icon="🤖", layout="centered")

# Caminhos dos artefatos salvos
diretorio_atual = Path(__file__).resolve().parent
caminho_modelo = diretorio_atual / 'modelo_regressao_logistica.pkl'
caminho_vetorizador = diretorio_atual / 'vetorizador_tfidf.pkl'

# Carregar o modelo e o vetorizador com cache para otimizar
@st.cache_resource
def carregar_artefatos():
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetorizador)
    return modelo, vetorizador

try:
    modelo, vetorizador = carregar_artefatos()
except Exception as e:
    st.error(f"Erro ao carregar os artefatos `.pkl`. Certifica-te de que estão na mesma pasta. Detalhe: {e}")
    st.stop()

# Interface Gráfica
st.title("📊 Análise de Sentimentos em E-commerce")
st.markdown("Projeto Final UC9 - SENAC Mato Grosso")
st.write("Digite uma avaliação de produto abaixo para que o modelo de **Regressão Logística** classifique o sentimento em tempo real.")

# Caixa de texto para o utilizador inserir a review
texto_usuario = st.text_area("Escreva a avaliação do cliente:", placeholder="Ex: O produto chegou super rápido, excede todas as expectativas e a qualidade é excelente!")

if st.button("Analisar Sentimento", type="primary"):
    try:
        if text_input := texto_usuario.strip():
            texto_lower = text_input.lower()
            
            # Lista ampliada de termos e padrões de insatisfação no e-commerce
            termos_negativos = [
                "não veio", "nao veio", "não gostei", "nao gostei", 
                "não funciona", "nao funciona", "não valeu", "nao valeu",
                "demorou", "péssimo", "horrível", "ruim", "decepcionou", 
                "estragado", "defeituoso", "fraude", "arrependi"
            ]
            
            # Verifica se algum termo crítico está presente na avaliação
            possui_insatisfacao = any(termo in texto_lower for termo in termos_negativos)

            st.divider()
            st.subheader("Resultado da Predição:")

            # Se houver termos claros de insatisfação, forçamos a análise crítica com nota técnica
            if possui_insatisfacao:
                st.error("🔴 **Sentimento Negativo / Insatisfeito** (Classificação Crítica por Tratamento de Negação/Insatisfação)")
                st.info("ℹ️ *Nota técnica:* Aplicada correção de rota heurística para mitigar falsos positivos do modelo linear TF-IDF em frases complexas.")
            else:
                # 1. Vetorizar o texto
                texto_vetorizado = vetorizador.transform([text_input])
                
                # 2. Previsão e Probabilidades
                predicao = modelo.predict(texto_vetorizado)[0]
                probabilidade = modelo.predict_proba(texto_vetorizado)[0]
                confianca = max(probabilidade) * 100
                
                # Exibir resultado padrão do modelo
                if predicao == 1:
                    st.success(f"🟢 **Sentimento Positivo** (Confiança: {confianca:.2f}%)")
                    st.balloons()
                else:
                    st.error(f"🔴 **Sentimento Negativo / Insatisfeito** (Confiança: {confianca:.2f}%)")
        else:
            st.warning("Por favor, digite um texto válido para análise.")
            
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processamento da predição: {e}")