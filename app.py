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
    if text_input := texto_usuario.strip():
        # 1. Vetorizar o texto usando o TfidfVectorizer treinado
        texto_vetorizado = vetorizador.transform([text_input])
        
        # 2. Fazer a previsão com a Regressão Logística
        predicao = modelo.predict(texto_vetorizado)[0]
        
        # 3. Obter probabilidades (confiança do modelo)
        probabilidade = modelo.predict_proba(texto_vetorizado)[0]
        confianca = max(probabilidade) * 100
        
        st.divider()
        st.subheader("Resultado da Predição:")
        
        # Exibir o resultado visualmente com base na polaridade (0 = Negativo/Neutro, 1 = Positivo)
        if predicao == 1:
            st.success(f"🟢 **Sentimento Positivo** (Confiança: {confianca:.2f}%)")
            st.balloons()
        else:
            st.error(f"🔴 **Sentimento Negativo / Insatisfeito** (Confiança: {confianca:.2f}%)")
    else:
        st.warning("Por favor, digite um texto válido para análise.")