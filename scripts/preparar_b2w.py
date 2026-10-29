"""Prepara data/b2w_preparado.csv a partir da B2W-Reviews01 original.

Nao treina nem substitui modelos. Uso (na raiz):

    python scripts/preparar_b2w.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "data" / "B2W-Reviews01.csv"
SAIDA = RAIZ / "data" / "b2w_preparado.csv"
AMOSTRA = RAIZ / "data" / "b2w_amostra.csv"
RESUMO = RAIZ / "data" / "b2w_resumo.json"


def rotular_nota(nota) -> str | None:
    try:
        n = int(nota)
    except (TypeError, ValueError):
        return None
    if n in (1, 2):
        return "Negativo"
    if n == 3:
        return "Neutro"
    if n in (4, 5):
        return "Positivo"
    return None


def main() -> None:
    if not ORIGEM.exists():
        raise SystemExit(
            f"Arquivo original nao encontrado: {ORIGEM}\n"
            "Baixe B2W-Reviews01.csv de "
            "https://github.com/americanas-tech/b2w-reviews01 "
            "e coloque em data/B2W-Reviews01.csv"
        )

    df = pd.read_csv(ORIGEM, low_memory=False)
    n_original = len(df)
    df["sentimento_rotulo"] = df["overall_rating"].map(rotular_nota)
    texto = df["review_text"].fillna("").astype(str).str.strip()
    mask = (texto != "") & df["sentimento_rotulo"].notna()
    prep = df.loc[mask, [
        "submission_date",
        "product_name",
        "product_brand",
        "site_category_lv1",
        "site_category_lv2",
        "review_title",
        "review_text",
        "overall_rating",
        "recommend_to_a_friend",
        "sentimento_rotulo",
    ]].copy()
    prep["product_brand"] = prep["product_brand"].fillna("")
    prep["site_category_lv2"] = prep["site_category_lv2"].fillna("")
    prep["review_title"] = prep["review_title"].fillna("")
    prep.to_csv(SAIDA, index=False)

    amostra = prep.sample(n=min(80, len(prep)), random_state=42)
    amostra.to_csv(AMOSTRA, index=False)

    dist = prep["sentimento_rotulo"].value_counts().to_dict()
    resumo = {
        "origem": "B2W-Reviews01 (Americanas.com, jan-mai/2018)",
        "repositorio_oficial": "https://github.com/americanas-tech/b2w-reviews01",
        "repositorio_historico": "https://github.com/b2wdigital/b2w-reviews01",
        "licenca": "CC BY-NC-SA 4.0",
        "credito": "B2W Digital",
        "arquivo_original": "data/B2W-Reviews01.csv",
        "arquivo_preparado": "data/b2w_preparado.csv",
        "n_original": int(n_original),
        "n_preparado": int(len(prep)),
        "n_removidos_sem_texto_ou_nota": int(n_original - len(prep)),
        "campos_originais": [
            "submission_date",
            "reviewer_id",
            "product_id",
            "product_name",
            "product_brand",
            "site_category_lv1",
            "site_category_lv2",
            "review_title",
            "overall_rating",
            "recommend_to_a_friend",
            "review_text",
            "reviewer_birth_year",
            "reviewer_gender",
            "reviewer_state",
        ],
        "campos_usados": {
            "texto_avaliacao": "review_text",
            "titulo": "review_title",
            "nome_produto": "product_name",
            "marca": "product_brand",
            "categoria": "site_category_lv1",
            "subcategoria": "site_category_lv2",
            "nota": "overall_rating",
            "recomenda": "recommend_to_a_friend",
        },
        "regra_rotulagem": {
            "Negativo": "overall_rating 1 ou 2",
            "Neutro": "overall_rating 3",
            "Positivo": "overall_rating 4 ou 5",
            "observacao": "Rotulo derivado da nota, nao e previsao do modelo. O modelo treinado permanece binario (positivo/negativo).",
        },
        "distribuicao_preparado": {k: int(v) for k, v in dist.items()},
        "percentual_preparado": {k: round(v / len(prep) * 100, 2) for k, v in dist.items()},
        "top_categorias": prep["site_category_lv1"].value_counts().head(10).to_dict(),
        "marca_preenchida": int((prep["product_brand"].astype(str).str.strip() != "").sum()),
        "marca_vazia": int((prep["product_brand"].astype(str).str.strip() == "").sum()),
    }
    RESUMO.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"original: {n_original}")
    print(f"preparado: {len(prep)} -> {SAIDA}")
    print(f"amostra: {len(amostra)} -> {AMOSTRA}")
    print(f"resumo -> {RESUMO}")


if __name__ == "__main__":
    main()
