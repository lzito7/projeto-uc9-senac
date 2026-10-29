# Publicar no Streamlit Community Cloud

Versão deste pacote: `app.py` com 7 páginas, tema escuro, TF-IDF + Regressão Logística,
heurística de negação desativada por padrão.

## 1. Arquivos que compõem a versão atual

```
app.py                    # aplicação (streamlit run app.py)
preprocessamento.py       # normalização usada no treino e na inferência
tema.py                   # CSS e componentes visuais
requirements.txt          # dependências
runtime.txt               # python-3.11
.python-version           # 3.11
.streamlit/config.toml    # tema escuro + toolbar minimal
models/modelo_regressao_logistica.pkl
models/vetorizador_tfidf.pkl
models/metricas.json
data/exemplo_avaliacoes.csv
data/b2w_amostra.csv
data/b2w_resumo.json
data/ORIGEM_B2W.md
scripts/treinar_modelo.py
scripts/diagnostico.py
scripts/preparar_b2w.py
README.md
.gitignore
```

Não versionar (grandes): `data/B2W-Reviews01.csv` (~47 MB), `data/b2w_preparado.csv`
(~37 MB), `data/df_clean.csv` (~715 mil linhas). Já estão no `.gitignore`.

## 2. Transferir para o GitHub

O repositório atual tem a versão antiga com `IndentationError` na linha 627.
O ideal é substituir o conteúdo mantendo o histórico, com um commit novo
(nada é apagado no histórico).

### Opção A - pelo navegador (sem git instalado)

1. Extraia o ZIP `projeto-uc9-senac-atualizado.zip`.
2. Abra https://github.com/lzito7/projeto-uc9-senac
3. Para cada arquivo da lista acima: abra o caminho correspondente no GitHub,
   clique em `Add file` > `Upload files`, arraste os arquivos e clique em
   `Commit changes` na branch `main`.
4. Se o GitHub recusar o upload de um arquivo existente, abra o arquivo no
   repositório, clique no ícone de lápis, apague o conteúdo, cole o novo
   conteúdo e faça o commit (a página de `app.py` tem 39 KB; cole o conteúdo
   do arquivo extraído do ZIP).
5. Confirme que `models/` contém os três artefatos e que `data/` contém apenas
   os arquivos pequenos.

### Opção B - com git (recomendado)

```bash
git clone https://github.com/lzito7/projeto-uc9-senac.git
cd projeto-uc9-senac
```

Copie para essa pasta os arquivos do ZIP, sobrescrevendo os existentes
(mantenha a pasta `.git`).

```bash
git status
git add app.py tema.py preprocessamento.py requirements.txt runtime.txt README.md .gitignore .streamlit/config.toml models data scripts
git commit -m "feat: publica versao atualizada do app UC9 (Streamlit)"
git push origin main
```

Se o push for rejeitado por existirem commits novos no remoto:

```bash
git pull --rebase origin main
git push origin main
```

Não use `git push --force`, para não perder o histórico existente.

## 3. Publicar no Streamlit Cloud

1. Acesse https://share.streamlit.io e entre com a conta GitHub.
2. `Create app` > `Deploy a public app from GitHub`.
3. Repository: `lzito7/projeto-uc9-senac`
4. Branch: `main`
5. Main file path: `app.py`
6. Python version: 3.11 (definida por `runtime.txt`)
7. `Deploy`.

O primeiro build instala o `requirements.txt` e leva alguns minutos. Depois de
cada `git push` na branch `main`, o app é reconstruído automaticamente.

## 4. Verificação pós-deploy

- A página inicial deve mostrar o hero e a barra lateral com 7 páginas.
- A página `Modelos` deve exibir acurácia 94,57%.
- A página `Relatórios` deve aceitar `data/exemplo_avaliacoes.csv`.
- Se aparecer erro de módulo, use `Manage app` > `Reboot app`.

## 5. Observação sobre versões

O `requirements.txt` usa faixas (`streamlit>=1.32,<2`, `scikit-learn>=1.3,<2`,
`pandas>=2.0,<4`, `joblib>=1.3`, `fpdf2>=2.7`). O modelo e o vetorizador foram
treinados e validados com scikit-learn 1.9.1 e carregam sem aviso de versão.
