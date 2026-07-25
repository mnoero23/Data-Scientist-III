# Pipeline de Preprocesamiento y Diagnóstico de Texto

Pre-entrega del Módulo 2 de **Data Science III: NLP & Deep Learning**.

El proyecto construye un pipeline reproducible de limpieza, normalización,
lematización y diagnóstico exploratorio sobre el corpus **AG News**. El
objetivo es dejar el conjunto de entrenamiento preparado para desarrollar un
clasificador TF-IDF y, posteriormente, comparar ese baseline con un Transformer
ajustado mediante LoRA.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Documentos de entrenamiento | 8.000 |
| Documentos de test reservados | 2.000 |
| Clases | 4 |
| Documentos por clase en train | 2.000 |
| Mediana de longitud | 23 tokens |
| Percentil 95 | 32 tokens |
| Percentil 99 | 42 tokens |
| Longitud máxima | 70 tokens |
| Documentos vacíos después del pipeline | 0 |

El conjunto de entrenamiento está perfectamente balanceado entre `World`,
`Sports`, `Business` y `Sci_Tech`. El percentil 95 sugiere
`max_len=32` como referencia inicial, aunque este valor deberá recalcularse con
el tokenizador WordPiece antes de entrenar el Transformer.

## Visualizaciones

### Distribución de longitud

![Distribución de longitud](reports/figures/distribucion_longitud.png)

### Distribución de clases

![Distribución de clases](reports/figures/distribucion_clases.png)

## Pipeline

1. Decodificación de entidades HTML.
2. Eliminación de tags HTML, URLs, dominios y correos.
3. Conversión a minúsculas y normalización de espacios.
4. Tokenización y lematización con SpaCy (`en_core_web_sm`).
5. Eliminación de puntuación, stop-words y boilerplate editorial.
6. Conservación deliberada de negaciones relevantes.
7. Cálculo de longitudes, frecuencias, bi-gramas, tri-gramas y balance de clases.

El análisis se ejecuta exclusivamente sobre `ag_news_train.csv`.
`ag_news_test.csv` permanece aislado para evitar **data leakage** y se utilizará
solo en la evaluación final de los modelos.

## Estructura

```text
ds3_eda_nlp_modulo2/
├── data/
│   └── README.md
├── reports/
│   ├── EDA_NLP_NoeroMatias.pdf
│   └── figures/
│       ├── distribucion_clases.png
│       └── distribucion_longitud.png
├── results/
│   ├── resumen_eda.csv
│   ├── top_20_bigramas.csv
│   ├── top_20_trigramas.csv
│   └── top_50_palabras.csv
├── src/
│   └── preprocess_eda.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Instalación

Se recomienda Python 3.10 o superior.

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

En macOS o Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset

Descargá los dos archivos de AG News indicados en
[`data/README.md`](data/README.md) y guardalos dentro de `data/`:

```text
data/ag_news_train.csv
data/ag_news_test.csv
```

El archivo de test no es leído por el script del EDA.

## Ejecución

Desde la raíz del repositorio:

```bash
python src/preprocess_eda.py
```

El proceso crea:

- corpus de entrenamiento preprocesado;
- resumen estadístico;
- top 50 de palabras;
- top 20 de bi-gramas y tri-gramas;
- histograma de longitud;
- gráfico de distribución de clases.

## Hallazgos

Los n-gramas capturan señales temáticas interpretables:

- `oil price`, `wall street` y `microsoft corp`: Business/Sci_Tech;
- `red sox` y `world cup`: Sports;
- `united states`, `prime minister` y `president bush`: World.

También se detectó boilerplate editorial residual, como
`quote profile research` y `press canadian press`. Se documenta para evaluar su
impacto mediante validación en el módulo siguiente, evitando eliminar términos
potencialmente útiles sin evidencia.

## Informe

El entregable final se encuentra en:

[`reports/EDA_NLP_NoeroMatias.pdf`](reports/EDA_NLP_NoeroMatias.pdf)

## Próximos pasos

- Ajustar `TfidfVectorizer` únicamente con train.
- Entrenar un clasificador supervisado baseline.
- Evaluar sobre el test reservado.
- Comparar el baseline contra un Transformer con LoRA sobre el mismo test.

## Autor

**Matías Noero**

