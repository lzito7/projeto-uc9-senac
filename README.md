# Projeto Final UC9 - Análise de Sentimentos em E-commerce (SENAC-MT)

Classificação automática de sentimento (positivo/negativo) em avaliações de produtos, usando TF-IDF e Regressão Logística, com interface em Streamlit.

## Equipe
- Luiz Dias Frederico Neto
- Bruno Kenzo Nobre Miyagawa
- João Lucas Fernandes de Barros

## Estrutura

```
projeto-uc9-senac/
├── app.py                  # aplicação Streamlit (streamlit run app.py)
├── tema.py                 # estilo visual
├── requirements.txt
├── .streamlit/config.toml  # tema escuro
├── models/                 # artefatos treinados
│   ├── modelo_regressao_logistica.pkl
│   ├── vetorizador_tfidf.pkl
│   └── metricas.json       # gerado pelo treino
├── data/
│   ├── exemplo_avaliacoes.csv   # 20 avaliações para testar o modo lote
│   └── df_clean.csv             # base de treino (NÃO versionada)
└── scripts/
    ├── treinar_modelo.py   # pipeline: treino, comparação de modelos, métricas
    └── diagnostico.py      # inspeção do modelo e do vocabulário
```

## Como executar

```bash
git clone https://github.com/lzito7/projeto-uc9-senac.git
cd projeto-uc9-senac
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Para testar o modo lote, envie `data/exemplo_avaliacoes.csv` na aba "Análise em Lote".

## Como retreinar

O arquivo `data/df_clean.csv` (~715 mil linhas, colunas `review_text_processed` e `polarity`) **não está no repositório** por causa do tamanho. Com ele em `data/`:

```bash
python scripts/treinar_modelo.py
```

O script compara um baseline (classe majoritária), Regressão Logística, Naive Bayes e SVM Linear, salva o modelo e o vetorizador em `models/` e grava as métricas reais em `models/metricas.json`, que o app exibe na aba de documentação.

## Dados

`df_clean.csv`: 227.762 avaliações em português (origem: B2W-Reviews01), com `polarity` binária: 76,3% positivas e 23,7% negativas. O arquivo tem ~715 mil linhas físicas porque muitas avaliações contêm quebras de linha.

## Metodologia

1. Pré-processamento: minúsculas e remoção de acentos (`preprocessamento.py`). Reproduz a coluna `review_text_processed` em 100% das linhas e é aplicado ao texto digitado no app.
2. Vetorização TF-IDF (até 5.000 termos, unigramas e bigramas).
3. Regressão Logística treinada com divisão estratificada 80/20 (`random_state=42`).

## Resultados (conjunto de teste, n = 45.553)

| Cenário | Acurácia | Recall (macro) | F1 (macro) |
|---|---|---|---|
| Baseline (sempre positivo) | 76,30% | n/d | n/d |
| **Modelo + pré-processamento do treino** | **94,57%** | 91,52% | 92,34% |
| Modelo sem o pré-processamento (texto cru, com acento) | 91,46% | 84,52% | 87,21% |
| Modelo + heurística de palavras negativas | 77,08% | 81,86% | 74,09% |

Principais conclusões: o texto precisa passar pelo mesmo pré-processamento do treino (sem ele, o modelo trata "não" e "péssimo" como palavras desconhecidas); e a heurística de palavras negativas piora o resultado, então fica desativada por padrão (opção experimental na barra lateral). Métricas oficiais em `models/metricas.json`.

## Limitações conhecidas

- O modelo é binário; não existe classe neutra.
- Não detecta ironia. Exemplo: "ótimo, chegou quebrado, parabéns" é classificado como positivo.
- Treinado com avaliações de e-commerce brasileiro; pode generalizar mal para outros domínios.
- A classe negativa é minoritária (23,7%), por isso a acurácia deve ser lida junto do baseline e do recall.
