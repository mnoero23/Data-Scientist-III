# Data Scientist III — NLP & Deep Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![Estado](https://img.shields.io/badge/Curso-Completado-2EA44F)](#proyecto-final)

Repositorio central de proyectos, pre-entregas y artefactos finales del curso **Data Scientist III**.

El recorrido construye una solución end-to-end: fundamentos de redes neuronales, preparación de un corpus real, baseline TF-IDF, fine-tuning eficiente de un Transformer con LoRA y comparación final sobre el mismo conjunto de prueba.

**[Comenzar con la guía rápida de instalación y ejecución](QUICKSTART.md)**

## Navegación

- [Recorrido por módulos](#recorrido-por-módulos)
- [Resultado comparativo](#resultado-comparativo)
- [Evidencia visual](#evidencia-visual)
- [Proyecto final](#proyecto-final)
- [Reproducibilidad](#reproducibilidad)
- [Estructura](#estructura-del-repositorio)
- [Tecnologías](#tecnologías)

## Recorrido por módulos

| Etapa | Proyecto | Objetivo | Resultado principal |
|---|---|---|---|
| Módulo 1 | [Pipeline base de Deep Learning](01_modulo1_pipeline_pytorch/) | Implementar entrenamiento y validación reproducibles con PyTorch sobre Iris | Accuracy de validación: **93,33%** |
| Módulo 2 | [Preprocesamiento y EDA de NLP](02_modulo2_eda_nlp/) | Limpiar, lematizar y diagnosticar AG News sin utilizar el test | P95: **32 tokens**; 4 clases balanceadas |
| Módulo 3 | [Clasificador supervisado con TF-IDF](03_modulo3_tfidf_baseline/) | Construir un baseline supervisado sobre el corpus preparado | F1 weighted: **0,8925** |
| Módulo 4 | [Transformer con LoRA](04_modulo4_distilbert_lora/) | Ajustar DistilBERT de forma eficiente sobre AG News | F1 weighted: **0,9070**; 1,095% entrenable |
| Módulo 5 | [Capstone y resumen integral](05_proyecto_final_capstone/) | Comparar TF-IDF y DistilBERT + LoRA sobre el mismo test | **+1,46 pp** de F1 weighted con LoRA |

## Resultado comparativo

Los dos clasificadores fueron evaluados sobre el mismo `ag_news_test.csv` de **2.000 noticias**, con 500 ejemplos por clase.

| Modelo | Precision macro | Recall macro | F1 macro | F1 weighted | Accuracy | Aciertos |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 0,8926 | 0,8925 | 0,8925 | 0,8925 | 0,8925 | 1.785 |
| DistilBERT + LoRA | **0,9076** | **0,9070** | **0,9070** | **0,9070** | **0,9070** | **1.814** |

LoRA logró **29 predicciones correctas adicionales** y mejoró el F1 weighted en **1,46 puntos porcentuales**, entrenando solamente **741.124 de 67.697.672 parámetros**.

## Evidencia visual

| Baseline TF-IDF + LinearSVC | DistilBERT + LoRA |
|---|---|
| ![Matriz de confusión del baseline TF-IDF](03_modulo3_tfidf_baseline/results/confusion_matrix.png) | ![Matriz de confusión de DistilBERT con LoRA](04_modulo4_distilbert_lora/results/confusion_matrix_lora.png) |
| [Ver métricas y errores del baseline](03_modulo3_tfidf_baseline/results/) | [Ver métricas e historial de LoRA](04_modulo4_distilbert_lora/results/) |

La mayor dificultad de ambos modelos se concentra en la separación entre **Business** y **Sci_Tech**, categorías que comparten vocabulario sobre empresas, productos y mercados tecnológicos.

## Proyecto final

- [Informe técnico final — Capstone NLP](05_proyecto_final_capstone/NLP_Capstone_Noero_Matias.pdf)
- [Resumen completo del curso](05_proyecto_final_capstone/Resumen_Curso_Data_Scientist_III_Noero_Matias.pdf)
- [Presentación visual](05_proyecto_final_capstone/Presentacion_Curso_Data_Scientist_III_Noero_Matias.pptx)
- [Notebook reproducible de DistilBERT + LoRA](04_modulo4_distilbert_lora/Noero_Matias_LoRA_AGNews.ipynb)

## Reproducibilidad

- **Semillas documentadas:** los experimentos fijan semillas para reducir variaciones.
- **Prevención de data leakage:** las transformaciones se ajustan exclusivamente con entrenamiento.
- **Test común:** TF-IDF y LoRA se comparan sobre la misma partición reservada.
- **Entornos independientes:** cada módulo mantiene sus propias dependencias.
- **Trazabilidad:** se conservan métricas, matrices, reportes, errores y configuración.
- **Interpretación responsable:** no se atribuyen mecanismos de atención que no hayan sido calculados.

Para ejecutar los proyectos, consultá [QUICKSTART.md](QUICKSTART.md).

## Estructura del repositorio

```text
Data-Scientist-III/
├── QUICKSTART.md
├── 01_modulo1_pipeline_pytorch/
├── 02_modulo2_eda_nlp/
├── 03_modulo3_tfidf_baseline/
├── 04_modulo4_distilbert_lora/
│   ├── Noero_Matias_LoRA_AGNews.ipynb
│   ├── Noero_Matias_LoRA_resultados.zip
│   ├── results/
│   └── README.md
├── 05_proyecto_final_capstone/
│   ├── NLP_Capstone_Noero_Matias.pdf
│   ├── Presentacion_Curso_Data_Scientist_III_Noero_Matias.pptx
│   ├── Resumen_Curso_Data_Scientist_III_Noero_Matias.pdf
│   └── README.md
└── README.md
```

## Tecnologías

| Área | Herramientas |
|---|---|
| Lenguaje y cómputo | Python, NumPy, pandas |
| Deep Learning | PyTorch |
| NLP clásico | SpaCy, TF-IDF, scikit-learn |
| Transformers | Hugging Face Transformers, PEFT, LoRA |
| Visualización | Matplotlib, Seaborn |
| Entornos | Google Colab, Jupyter |

## Autor

**Matías Noero**  
Analista de datos en formación continua hacia Data Science, NLP y Deep Learning.
