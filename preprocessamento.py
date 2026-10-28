import unicodedata


def normalizar_texto(texto) -> str:
    """Mesmo tratamento do treino: minúsculas e sem acentos.

    Reproduz a coluna `review_text_processed` do df_clean.csv (verificado em 100% das linhas).
    """
    t = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in t if not unicodedata.combining(c))
