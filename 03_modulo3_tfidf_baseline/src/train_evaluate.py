"""
Pre-entrega Módulo 3 - Clasificador supervisado con TF-IDF.

Dataset: AG News provisto por la cátedra.
El test permanece aislado de la selección de hiperparámetros.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
CLASS_ORDER = ["World", "Sports", "Business", "Sci_Tech"]
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "ag_news_train.csv"
DEFAULT_TEST_PATH = BASE_DIR / "data" / "ag_news_test.csv"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_MODELS_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int = 2
    max_df: float = 0.98
    sublinear_tf: bool = True


CANDIDATES = [
    CandidateConfig("uni_20k", 20_000, (1, 1)),
    CandidateConfig("uni_bi_20k", 20_000, (1, 2)),
    CandidateConfig("uni_40k", 40_000, (1, 1)),
    CandidateConfig("uni_bi_40k", 40_000, (1, 2)),
]


def clean_text(text: str) -> str:
    """Limpia HTML, URLs, correos y ruido sin utilizar información del test."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b(?:[\w-]+\.)+(?:com|org|net|gov|edu)\b", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """Carga y valida uno de los splits oficiales de AG News."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descargá el archivo indicado en data/README.md."
        )

    frame = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{split_name} debe contener las columnas {sorted(required)}; "
            f"se encontraron {list(frame.columns)}."
        )
    if frame[["text", "label"]].isna().any().any():
        raise ValueError(f"{split_name} contiene nulos en text o label.")
    if set(frame["label"].unique()) != set(CLASS_ORDER):
        raise ValueError(
            f"Etiquetas inesperadas en {split_name}: "
            f"{sorted(frame['label'].unique())}."
        )
    if (frame["text"].astype(str).str.strip() == "").any():
        raise ValueError(f"{split_name} contiene textos vacíos.")
    return frame[["text", "label"]].copy()


def build_vectorizer(config: CandidateConfig) -> TfidfVectorizer:
    """Crea un vectorizador con la configuración indicada."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        strip_accents="unicode",
        stop_words="english",
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        norm="l2",
        dtype=np.float32,
    )


def select_configuration(train_df: pd.DataFrame) -> tuple[CandidateConfig, pd.DataFrame]:
    """Selecciona parámetros con una validación estratificada dentro de train."""
    inner_train, validation = train_test_split(
        train_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_df["label"],
    )

    rows: list[dict[str, object]] = []
    for config in CANDIDATES:
        vectorizer = build_vectorizer(config)
        x_inner = vectorizer.fit_transform(inner_train["text"])
        x_validation = vectorizer.transform(validation["text"])

        model = LinearSVC(C=1.0, random_state=RANDOM_STATE)
        model.fit(x_inner, inner_train["label"])
        predictions = model.predict(x_validation)

        rows.append(
            {
                **asdict(config),
                "ngram_range": str(config.ngram_range),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "validation_accuracy": accuracy_score(
                    validation["label"], predictions
                ),
                "validation_macro_f1": f1_score(
                    validation["label"], predictions, average="macro"
                ),
                "validation_weighted_f1": f1_score(
                    validation["label"], predictions, average="weighted"
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=False,
    )
    best_name = str(results.iloc[0]["name"])
    best_config = next(config for config in CANDIDATES if config.name == best_name)
    return best_config, results.reset_index(drop=True)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    """Guarda la matriz de confusión con conteos y porcentajes por clase real."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            annotations[row, column] = (
                f"{matrix[row, column]}\n{percentages[row, column]:.1%}"
            )

    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        linewidths=0.5,
        ax=ax,
    )
    ax.set(
        title="Matriz de confusión - AG News test",
        xlabel="Clase predicha",
        ylabel="Clase real",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def hardest_confusions(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Resume los pares de clases con mayor cantidad de confusiones."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    rows: list[dict[str, object]] = []
    for true_index, true_label in enumerate(CLASS_ORDER):
        for pred_index, predicted_label in enumerate(CLASS_ORDER):
            if true_index != pred_index:
                rows.append(
                    {
                        "real": true_label,
                        "predicha": predicted_label,
                        "errores": int(matrix[true_index, pred_index]),
                    }
                )
    return pd.DataFrame(rows).sort_values("errores", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline TF-IDF + LinearSVC para AG News."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.train, "train")
    test_df = load_split(args.test, "test")

    best_config, candidate_results = select_configuration(train_df)
    candidate_results.to_csv(
        args.results_dir / "candidate_results.csv", index=False
    )

    # Ajuste final: fit solo sobre el train oficial.
    vectorizer = build_vectorizer(best_config)
    x_train = vectorizer.fit_transform(train_df["text"])

    # Evaluación final: transform, nunca fit, sobre el test oficial.
    x_test = vectorizer.transform(test_df["text"])

    model = LinearSVC(C=1.0, random_state=RANDOM_STATE)
    model.fit(x_train, train_df["label"])
    predictions = model.predict(x_test)

    report_dict = classification_report(
        test_df["label"],
        predictions,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        test_df["label"],
        predictions,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        digits=4,
        zero_division=0,
    )

    report_frame = pd.DataFrame(report_dict).transpose()
    report_frame.to_csv(args.results_dir / "classification_report.csv")
    (args.results_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    confusion_pairs = hardest_confusions(test_df["label"], predictions)
    confusion_pairs.to_csv(args.results_dir / "confusion_pairs.csv", index=False)

    save_confusion_matrix(
        test_df["label"],
        predictions,
        args.results_dir / "confusion_matrix.png",
    )

    errors = test_df.loc[test_df["label"].to_numpy() != predictions].copy()
    errors["predicted_label"] = predictions[
        test_df["label"].to_numpy() != predictions
    ]
    errors.head(100).to_csv(
        args.results_dir / "misclassified_samples.csv", index=False
    )

    metrics = {
        "random_state": RANDOM_STATE,
        "train_documents": len(train_df),
        "test_documents": len(test_df),
        "classes": CLASS_ORDER,
        "selected_config": {
            **asdict(best_config),
            "ngram_range": list(best_config.ngram_range),
        },
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_matrix_shape": list(x_train.shape),
        "test_matrix_shape": list(x_test.shape),
        "test_accuracy": accuracy_score(test_df["label"], predictions),
        "test_macro_f1": f1_score(
            test_df["label"], predictions, average="macro"
        ),
        "test_weighted_f1": f1_score(
            test_df["label"], predictions, average="weighted"
        ),
        "test_errors": int((test_df["label"].to_numpy() != predictions).sum()),
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    joblib.dump(vectorizer, args.models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, args.models_dir / "linear_svc.joblib")

    print("\nConfiguraciones evaluadas solo dentro de train:")
    print(candidate_results.to_string(index=False))
    print(f"\nConfiguración seleccionada: {best_config.name}")
    print("\nEvaluación final sobre test:")
    print(report_text)
    print("Mayores confusiones:")
    print(confusion_pairs.head(8).to_string(index=False))
    print("\nMétricas finales:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

