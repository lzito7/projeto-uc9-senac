"""Diagnóstico do modelo. Uso (na raiz do projeto): python scripts/diagnostico.py"""
import warnings
from pathlib import Path
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent

df = pd.read_csv(RAIZ / "data" / "df_clean.csv", on_bad_lines="skip", nrows=50000)
print("Colunas:", df.columns.tolist())
print("\nDistribuição de polarity:\n", df["polarity"].value_counts(normalize=True).round(3))
print("\n10 exemplos de texto processado:")
print(df["review_text_processed"].dropna().sample(10, random_state=1).to_string())

v = joblib.load(RAIZ / "models" / "vetorizador_tfidf.pkl")
m = joblib.load(RAIZ / "models" / "modelo_regressao_logistica.pkl")
vocab = list(v.get_feature_names_out())
print("\n30 termos do vocabulário:", vocab[:30])
print("Termos com acento:", sum(any(c in t for c in "áéíóúãõçâêô") for t in vocab), "de", len(vocab))
for termo in ["nao", "não", "pessimo", "péssimo"]:
    print(f"'{termo}' no vocabulário:", termo in vocab)

print("\nMesma frase, crua vs. formato do treino:")
for t in ["não gostei, péssimo", "nao gostei pessimo", "adorei o produto", "produto horrível"]:
    x = v.transform([t])
    print(f"{t!r} -> {m.predict(x)[0]} ({max(m.predict_proba(x)[0]) * 100:.1f}%)")
