# Data Scientist III - NLP & Deep Learning

Repositorio central de proyectos, pre-entregas y artefactos finales del curso **Data Scientist III**.

El recorrido construye una solución end-to-end: fundamentos de redes neuronales, preparación de un corpus real, baseline TF-IDF, fine-tuning eficiente de un Transformer con LoRA y comparación final sobre el mismo conjunto de prueba.

**[Comenzar con la guía rápida de instalación y ejecución](QUICKSTART.md)**

## Proyectos

| Etapa | Proyecto | Objetivo | Resultado principal |
|---|---|---|---|
| Módulo 1 | [Pipeline base de Deep Learning](01_modulo1_pipeline_pytorch/) | Implementar entrenamiento y validación reproducibles con PyTorch sobre Iris | Accuracy de validación: **93,33%** |
| Módulo 2 | [Preprocesamiento y EDA de NLP](02_modulo2_eda_nlp/) | Limpiar, lematizar y diagnosticar AG News sin usar el test | P95: **32 tokens**; 4 clases balanceadas |
| Módulo 3 | [Clasificador supervisado con TF-IDF](03_modulo3_tfidf_baseline/) | Construir un baseline supervisado sobre el corpus del Módulo 2 | F1 weighted: **0,8925** |
| Módulo 4 | [Transformer con LoRA](04_modulo4_distilbert_lora/) | Ajustar DistilBERT de forma eficiente sobre AG News | F1 weighted: **0,9070**; 1,095% entrenable |
| Proyecto final | [Capstone y resumen integral](05_proyecto_final_capstone/) | Comparar TF-IDF y DistilBERT + LoRA sobre el mismo test | **+1,46 pp** de F1 weighted con LoRA |

## Descargables finales

- [Resumen completo del curso en PDF](05_proyecto_final_capstone/Resumen_Curso_Data_Scientist_III_Noero_Matias.pdf)
- [Presentación visual del curso](05_proyecto_final_capstone/Presentacion_Curso_Data_Scientist_III_Noero_Matias.pptx)
- [Informe técnico final - Capstone NLP](05_proyecto_final_capstone/NLP_Capstone_Noero_Matias.pdf)

## Resultado comparativo

Los dos modelos fueron evaluados sobre el mismo `ag_news_test.csv` de 2.000 noticias, con 500 ejemplos por clase.

| Modelo | Precision macro | Recall macro | F1 macro | F1 weighted | Accuracy |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 0,8926 | 0,8925 | 0,8925 | 0,8925 | 0,8925 |
| DistilBERT + LoRA | 0,9076 | 0,9070 | 0,9070 | 0,9070 | 0,9070 |

LoRA obtuvo **1.814/2.000** predicciones correctas, frente a **1.785/2.000** del baseline.

## Proyectos por módulo

### Módulo 1 - Pipeline de entrenamiento y validación

Red neuronal multicapa en PyTorch con split estratificado, escalado ajustado exclusivamente con train, semillas reproducibles y ciclos separados de entrenamiento y validación.

[Ver proyecto y código](01_modulo1_pipeline_pytorch/)

### Módulo 2 - Preprocesamiento y diagnóstico de NLP

Pipeline sobre AG News con limpieza Regex, normalización, tokenización, lematización con SpaCy, frecuencias, n-gramas, longitudes y balance de clases.

[Ver proyecto, código e informe](02_modulo2_eda_nlp/)

### Módulo 3 - Clasificador TF-IDF

Baseline multiclase con `TfidfVectorizer` y `LinearSVC`. La selección de parámetros se realizó dentro de train para evitar data leakage.

[Ver proyecto, código y resultados](03_modulo3_tfidf_baseline/)

### Módulo 4 - DistilBERT + LoRA

Fine-tuning eficiente con r=8, alpha=16, dropout=0,1, target modules `q_lin`/`v_lin`, learning rate 2e-4 y 3 épocas. Se entrenaron 741.124 de 67.697.672 parámetros.

[Ver notebook, configuración y resultados](04_modulo4_distilbert_lora/)

## Principios del proyecto

- **Reproducibilidad:** semillas, parámetros y resultados documentados.
- **Prevención de data leakage:** transformaciones ajustadas únicamente con train.
- **Comparabilidad:** mismo conjunto de test para ambos modelos finales.
- **Trazabilidad:** métricas, matrices, informes y limitaciones conservados.
- **Interpretación responsable:** no se afirma evidencia de atención que no fue calculada.

## Tecnologías

- Python y PyTorch
- pandas y NumPy
- scikit-learn
- SpaCy
- Matplotlib y Seaborn
- Hugging Face Transformers
- PEFT / LoRA

## Estructura del repositorio

```text
Data-Scientist-III/
├── QUICKSTART.md
├── 01_modulo1_pipeline_pytorch/
├── 02_modulo2_eda_nlp/
├── 03_modulo3_tfidf_baseline/
├── 04_modulo4_distilbert_lora/
│   ├── results/
│   ├── Noero_Matias_LoRA_resultados.zip
│   └── README.md
├── 05_proyecto_final_capstone/
│   ├── README.md
│   ├── Resumen_Curso_Data_Scientist_III_Noero_Matias.pdf
│   ├── Presentacion_Curso_Data_Scientist_III_Noero_Matias.pptx
│   └── NLP_Capstone_Noero_Matias.pdf
└── README.md
```

## Autor

**Matías Noero**  
Analista de datos en formación continua hacia Data Science, NLP y Deep Learning.
