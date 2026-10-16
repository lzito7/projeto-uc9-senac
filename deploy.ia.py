# ==============================================================================
# PIPELINE COMPLETO: CARREGAMENTO, TREINO, 3 ALGORITMOS + BASELINE E EXPORTAÇÃO
# ==============================================================================
import pandas as pd
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("="*70)
print("--> [PASSO 1] CARREGANDO OS DADOS LOCALMENTE NO VS CODE")
print("="*70)

nome_arquivo_csv = 'df_clean.csv'
arquivo_csv = Path(__file__).resolve().parent / nome_arquivo_csv

try:
    df_clean = pd.read_csv(arquivo_csv, on_bad_lines='skip')
    print(f"Sucesso! Arquivo '{nome_arquivo_csv}' carregado.")
    print("Colunas detetadas no ficheiro:", df_clean.columns.tolist())
except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_csv}' não foi encontrado em '{arquivo_csv.parent}'.")
    raise SystemExit(1)

X_coluna = 'review_text_processed' 
y_coluna = 'polarity'

df_clean = df_clean.dropna(subset=[X_coluna, y_coluna]).copy()
df_clean[y_coluna] = df_clean[y_coluna].astype(int)

X = df_clean[X_coluna]
y = df_clean[y_coluna]

# Divisão estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("="*70)
print("--> [PASSO 2] VETORIZAÇÃO TF-IDF (Com Bigramas para capturar negações)")
print("="*70)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("="*70)
print("--> [PASSO 3] TREINAMENTO E COMPARAÇÃO (BASELINE + 3 ALGORITMOS)")
print("="*70)

modelos = {
    "0. Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
    "1. Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "2. Naive Bayes (MultinomialNB)": MultinomialNB(),
    "3. Support Vector Machine (LinearSVC)": LinearSVC(random_state=42, max_iter=2000)
}

resultados = []
modelos_treinados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_tfidf, y_train)
    modelos_treinados[nome] = modelo
    
    y_pred = modelo.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro') * 100
    
    resultados.append({
        "Modelo / Algoritmo": nome,
        "Acurácia (%)": round(acc, 2),
        "Macro F1-Score (%)": round(f1_macro, 2)
    })

df_resultado_comparativo = pd.DataFrame(resultados)
print("\n" + "="*70)
print(" TABELA COMPARATIVA OFICIAL (EXIGÊNCIA DO SENAC)")
print("="*70)
print(df_resultado_comparativo.to_string(index=False))
print("="*70)

print("="*70)
print("--> [PASSO 4] SALVANDO O MODELO CAMPEÃO PARA O DEPLOY")
print("="*70)

modelo_campeao = modelos_treinados["1. Regressão Logística"]

diretorio_script = Path(__file__).resolve().parent
caminho_modelo = os.path.join(diretorio_script, 'modelo_regressao_logistica.pkl')
caminho_vetor = os.path.join(diretorio_script, 'vetorizador_tfidf.pkl')

joblib.dump(modelo_campeao, caminho_modelo)
joblib.dump(vectorizer, caminho_vetor)

print(f"Ficheiro do modelo salvo em: {caminho_modelo}")
print(f"Ficheiro do vetorizador salvo em: {caminho_vetor}")