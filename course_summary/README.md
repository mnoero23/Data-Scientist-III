# Resumen completo - Data Scientist III

Este directorio reúne los artefactos de cierre del curso **Data Scientist III - NLP & Deep Learning**.

## Descargables

- [Resumen integral del curso en PDF](Resumen_Curso_Data_Scientist_III_Noero_Matias.pdf)
- [Presentación visual del curso](Presentacion_Curso_Data_Scientist_III_Noero_Matias.pptx)
- [Informe técnico final - Capstone NLP](NLP_Capstone_Noero_Matias.pdf)

## Recorrido

| Módulo | Tema | Producto principal |
|---|---|---|
| 1 | Fundamentos de Deep Learning | Pipeline PyTorch con entrenamiento y validación |
| 2 | NLP y preparación de texto | Corpus AG News limpio y diagnosticado |
| 3 | Modelos clásicos para NLP | TF-IDF + LinearSVC |
| 4 | Transformers y LoRA | DistilBERT ajustado eficientemente |
| 5 | Proyecto final | Comparación consolidada y defendible |

## Resultado comparativo

| Modelo | F1 weighted | Accuracy | Aciertos |
|---|---:|---:|---:|
| TF-IDF + LinearSVC | 0,8925 | 0,8925 | 1.785/2.000 |
| DistilBERT + LoRA | 0,9070 | 0,9070 | 1.814/2.000 |

Ambos modelos fueron evaluados sobre el mismo `ag_news_test.csv` de 2.000 noticias. LoRA mejoró el F1 weighted en **1,46 puntos porcentuales** y obtuvo **29 aciertos adicionales**.

## Autor

**Matías Noero**  
Analista de datos en formación continua hacia Data Science, NLP y Deep Learning.
