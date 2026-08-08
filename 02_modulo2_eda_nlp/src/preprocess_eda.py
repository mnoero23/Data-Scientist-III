"""
Pre-entrega Módulo 2 - Pipeline de Preprocesamiento y Diagnóstico (PPD)
Dataset: AG News provisto por la cátedra.

El análisis se ajusta exclusivamente sobre ag_news_train.csv.
ag_news_test.csv queda reservado para la evaluación final de los modelos.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from spacy.cli import download


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ag_news_train.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODEL_NAME = "en_core_web_sm"


def load_spacy_model(model_name: str = MODEL_NAME):
    """Carga SpaCy y descarga el modelo únicamente si no está instalado."""
    try:
        return spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        print(f"Descargando el modelo de SpaCy {model_name!r}...")
        download(model_name)
        return spacy.load(model_name, disable=["ner", "parser"])


def clean_text(text: str) -> str:
    """Elimina HTML, URLs y ruido estructural sin tocar el split ni la etiqueta."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b(?:[\w-]+\.)+(?:com|org|net|gov|edu)\b", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def preprocess_batch(texts: pd.Series, nlp, batch_size: int = 256) -> list[str]:
    """
    Limpia y lematiza con SpaCy.

    Se conservan negaciones como 'no', 'not' y 'never' porque pueden aportar
    significado. Se eliminan puntuación, espacios y stop-words generales.
    """
    cleaned = (clean_text(text) for text in texts)
    preserved_negations = {"no", "not", "never", "nor"}
    # Marcas de agencias repetidas en los copetes; funcionan como boilerplate
    # editorial y no aportan información temática a la clasificación.
    domain_noise = {"ap", "afp", "reuters", "reuter", "quot"}
    processed: list[str] = []

    for doc in nlp.pipe(cleaned, batch_size=batch_size):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_space
            and not token.is_punct
            and (not token.is_stop or token.lower_ in preserved_negations)
            and token.lemma_.lower() not in domain_noise
            and len(token.lemma_) > 1
        ]
        processed.append(" ".join(tokens))

    return processed


def top_ngrams(corpus: pd.Series, n: int, top_k: int = 20) -> pd.DataFrame:
    """Obtiene los n-gramas más frecuentes del corpus ya preprocesado."""
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=2)
    matrix = vectorizer.fit_transform(corpus)
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = frequencies.argsort()[::-1][:top_k]
    return pd.DataFrame(
        {"n_grama": terms[order], "frecuencia": frequencies[order].astype(int)}
    )


def save_plots(df: pd.DataFrame, p95: int) -> None:
    """Guarda las visualizaciones exigidas por la consigna."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(df["token_count"], bins=35, color="#3568A8", ax=ax)
    ax.axvline(
        p95,
        color="#D64541",
        linestyle="--",
        linewidth=2,
        label=f"Percentil 95 = {p95} tokens",
    )
    ax.set(
        title="Distribución de longitud del corpus preprocesado",
        xlabel="Cantidad de tokens por documento",
        ylabel="Cantidad de documentos",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribucion_longitud.png", dpi=180)
    plt.close(fig)

    class_order = ["World", "Sports", "Business", "Sci_Tech"]
    counts = df["label"].value_counts().reindex(class_order)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#3568A8", ax=ax)
    ax.set(
        title="Distribución de clases en AG News - train",
        xlabel="Categoría",
        ylabel="Cantidad de documentos",
    )
    for index, value in enumerate(counts.values):
        ax.text(index, value + 25, f"{value:,}".replace(",", "."), ha="center")
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribucion_clases.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas {sorted(required)}")

    if df[list(required)].isna().any().any():
        raise ValueError("El corpus contiene valores nulos en text o label.")

    nlp = load_spacy_model()
    df["processed_text"] = preprocess_batch(df["text"], nlp)
    df["token_count"] = df["processed_text"].str.split().str.len()

    p95 = int(np.ceil(df["token_count"].quantile(0.95)))
    median = float(df["token_count"].median())
    empty_docs = int((df["token_count"] == 0).sum())

    bigrams = top_ngrams(df["processed_text"], n=2)
    trigrams = top_ngrams(df["processed_text"], n=3)

    all_tokens = [
        token
        for document in df["processed_text"]
        for token in document.split()
    ]
    top_words = pd.DataFrame(
        Counter(all_tokens).most_common(50),
        columns=["palabra", "frecuencia"],
    )
    top_words["es_stopword"] = top_words["palabra"].isin(nlp.Defaults.stop_words)

    save_plots(df, p95)
    df[["text", "label", "processed_text", "token_count"]].to_csv(
        RESULTS_DIR / "ag_news_train_preprocesado.csv",
        index=False,
    )
    bigrams.to_csv(RESULTS_DIR / "top_20_bigramas.csv", index=False)
    trigrams.to_csv(RESULTS_DIR / "top_20_trigramas.csv", index=False)
    top_words.to_csv(RESULTS_DIR / "top_50_palabras.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "documentos_train": len(df),
                "clases": df["label"].nunique(),
                "mediana_tokens": median,
                "percentil_95_tokens": p95,
                "documentos_vacios": empty_docs,
                "stopwords_en_top_50": int(top_words["es_stopword"].sum()),
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "resumen_eda.csv", index=False)

    print(summary.to_string(index=False))
    print("\nTop 20 bi-gramas:\n", bigrams.to_string(index=False))
    print("\nTop 20 tri-gramas:\n", trigrams.to_string(index=False))


if __name__ == "__main__":
    main()
