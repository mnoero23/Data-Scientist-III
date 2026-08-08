# Clasificador supervisado con TF-IDF — Módulo 3

Pre-entrega de **Data Scientist III — NLP & Deep Learning**.

**Alumno:** Matías Noero  
**Dataset:** AG News  
**Objetivo:** construir un baseline reproducible de clasificación multiclase
sin fuga de datos.

## Enfoque

El proyecto transforma noticias crudas en vectores TF-IDF y entrena un
clasificador `LinearSVC`. Se eligió este modelo porque los SVM lineales suelen
ofrecer un baseline fuerte y eficiente en espacios de texto esparsos y de alta
dimensionalidad.

La selección de parámetros se realiza mediante una validación estratificada
interna sobre `ag_news_train.csv`. Se comparan:

- `max_features`: 20.000 y 40.000;
- `ngram_range`: unigramas `(1, 1)` y unigramas + bigramas `(1, 2)`.

Una vez elegida la configuración con mayor F1 Macro de validación, el
vectorizador y el modelo se reajustan con todo el train oficial. El test se
transforma con `transform` y se utiliza una sola vez para la evaluación final.

## Prevención de data leakage

1. Los parámetros se seleccionan solo dentro del split de entrenamiento.
2. `TfidfVectorizer.fit_transform()` se ejecuta solo sobre train.
3. `TfidfVectorizer.transform()` se ejecuta sobre test.
4. El vocabulario, los pesos IDF y el modelo nunca se ajustan con test.

## Estructura

```text
ds3_tfidf_baseline_modulo3/
├── data/
│   └── README.md
├── models/                         # artefactos locales, no versionados
├── results/
│   ├── candidate_results.csv
│   ├── classification_report.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── confusion_pairs.csv
│   ├── metrics.json
│   └── misclassified_samples.csv
├── src/
│   └── train_evaluate.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Ejecución

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python src/train_evaluate.py
```

En macOS o Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/train_evaluate.py
```

Los CSV deben descargarse siguiendo [`data/README.md`](data/README.md).

## Evaluación generada

El pipeline produce:

- `classification_report` con Precision, Recall y F1 por clase;
- Accuracy, F1 Macro y F1 Weighted;
- matriz de confusión multiclase;
- resumen de los pares de clases con mayor confusión;
- ejemplos de errores para análisis cualitativo;
- modelos serializados localmente con Joblib.

## Resultados obtenidos

Se entrenó con **8.000 noticias** y se evaluó una sola vez sobre **2.000
noticias de test**, con 500 ejemplos por categoría. La mejor configuración en
validación fue `max_features=40000` y `ngram_range=(1, 2)`.

| Configuración | Accuracy validación | F1 Macro validación |
|---|---:|---:|
| Unigramas + bigramas, 40k | **0,9019** | **0,9017** |
| Unigramas + bigramas, 20k | 0,9006 | 0,9005 |
| Unigramas, 20k | 0,8950 | 0,8947 |
| Unigramas, 40k | 0,8950 | 0,8947 |

### Métricas finales sobre test

| Clase | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| World | 0,9072 | 0,8800 | 0,8934 | 500 |
| Sports | 0,9508 | 0,9660 | 0,9583 | 500 |
| Business | 0,8353 | 0,8520 | 0,8436 | 500 |
| Sci_Tech | 0,8773 | 0,8720 | 0,8746 | 500 |
| **Promedio macro** | **0,8926** | **0,8925** | **0,8925** | **2.000** |

- **Accuracy:** 0,8925.
- **F1 Macro:** 0,8925.
- **F1 Weighted:** 0,8925.
- **Predicciones incorrectas:** 215 de 2.000.

## Interpretación

La principal dificultad fue separar **Business** de **Sci_Tech**: hubo 44 casos
de Business clasificados como Sci_Tech y otros 44 en el sentido inverso. Esto
es coherente con noticias sobre empresas tecnológicas, productos y mercados,
que comparten vocabulario de ambas categorías. También se observaron 35 casos
de World predichos como Business. **Sports** fue la categoría más fácil, con un
F1 de 0,9583, debido a su vocabulario más distintivo.

La evaluación es un baseline: no se realizó ajuste de hiperparámetros usando el
test ni se atribuye causalidad a los errores. Como mejora futura se propone
analizar los 215 ejemplos mal clasificados y comparar contra regresión
logística o modelos basados en embeddings.

## Reproducibilidad

- Semilla: `42`.
- Versiones exactas: `requirements.txt`.
- Los datos no se incluyen en GitHub; se mantienen los enlaces oficiales.
