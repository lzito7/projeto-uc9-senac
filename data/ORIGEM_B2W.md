# B2W-Reviews01 — origem, licença e rotulagem

## Origem

- Corpus aberto de avaliações de produtos da Americanas.com (janeiro a maio de 2018).
- Repositório oficial (arquivo): https://github.com/americanas-tech/b2w-reviews01
- Endereço histórico citado na licença: https://github.com/b2wdigital/b2w-reviews01
- Arquivo baixado e verificado: `data/B2W-Reviews01.csv` (49.453.175 bytes, 132.373 linhas, 14 colunas).
- O download foi feito de `https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv`.

## Licença

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

Uso acadêmico/não comercial, com crédito a **B2W Digital**, e obras derivadas sob a mesma licença.

https://creativecommons.org/licenses/by-nc-sa/4.0/

## Campos identificados

| Papel | Coluna |
|---|---|
| Texto da avaliação | `review_text` |
| Título | `review_title` |
| Nome do produto | `product_name` |
| Marca | `product_brand` (muitos vazios) |
| Categoria | `site_category_lv1` |
| Subcategoria | `site_category_lv2` |
| Nota (1 a 5) | `overall_rating` |
| Recomenda a um amigo | `recommend_to_a_friend` |

## Arquivos gerados (a original foi preservada)

- `data/B2W-Reviews01.csv` — original, não versionar (grande).
- `data/b2w_preparado.csv` — texto não vazio + rótulo de sentimento, não versionar (grande).
- `data/b2w_amostra.csv` — 80 linhas aleatórias (`random_state=42`), versionável.
- `data/b2w_resumo.json` — totais e metadados, versionável.

Regenerar:

```bash
python scripts/preparar_b2w.py
```

## Regra de rotulagem (nota, não é o modelo)

O modelo salvo em `models/` continua binário (positivo/negativo). O rótulo de três classes abaixo vem só da nota e serve para explorar o corpus:

- **Negativo**: `overall_rating` 1 ou 2
- **Neutro**: `overall_rating` 3
- **Positivo**: `overall_rating` 4 ou 5

Linhas sem `review_text` foram excluídas do arquivo preparado (3.275 de 132.373). Restaram 129.098 avaliações: 79.316 positivas (61,44%), 33.772 negativas (26,16%), 16.010 neutras (12,40%).
