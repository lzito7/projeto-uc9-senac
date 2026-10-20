# Projeto Final UC9 - Análise de Sentimentos em E-commerce (SENAC-MT)

Repositório oficial do projeto de Inteligência Artificial para classificação automática de sentimentos em avaliações de produtos utilizando Machine Learning.

## 👥 Equipe
* Luis Dias Frederico Neto
* Bruno Kenzo Nobre Miyagawa
* João Lucas Fernandes de Barros

---

## 🚀 Tecnologias Utilizadas
* **Python** (Processamento de Dados e Modelagem)
* **Scikit-Learn** (Regressão Logística e TF-IDF)
* **Streamlit** (Interface Gráfica para o Mini Deploy)
* **Joblib** (Serialização de Modelos)

---

## 📂 Estrutura do Repositório
* `app.py`: Código da interface web interativa em Streamlit.
* `deploy.ia.py`: Script principal de pipeline, limpeza, treino e validação de modelos.
* `modelo_regressao_logistica.pkl`: Artefato treinado do modelo campeão (Regressão Logística - 94.57% de acurácia).
* `vetorizador_tfidf.pkl`: Artefato do vetorizador de texto TF-IDF.

---

## ⚙️ Instruções de Execução (Reproducibilidade)

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/lzito7/projeto-uc9-senac.git](https://github.com/lzito7/projeto-uc9-senac.git)
   cd projeto-uc9-senac