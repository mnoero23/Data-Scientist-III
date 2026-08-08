# Guía rápida de ejecución

Esta guía permite recorrer y reproducir los proyectos del curso **Data Scientist III — NLP & Deep Learning** en orden cronológico.

## 1. Requisitos previos

- Git.
- Python 3.10 o superior.
- Conexión a Internet para instalar dependencias y descargar AG News.
- Google Colab con GPU para el Módulo 4.

> Cada módulo mantiene sus propias dependencias. Para evitar conflictos, se recomienda crear un entorno virtual independiente dentro de cada carpeta.

## 2. Descargar el repositorio

```bash
git clone https://github.com/mnoero23/Data-Scientist-III.git
cd Data-Scientist-III
```

## 3. Orden recomendado

| Orden | Proyecto | Forma de ejecución | Tiempo orientativo |
|---:|---|---|---:|
| 1 | Pipeline PyTorch sobre Iris | Local, CPU | 1–3 min |
| 2 | Preprocesamiento y EDA de AG News | Local, CPU | 10–20 min |
| 3 | Baseline TF-IDF + LinearSVC | Local, CPU | 2–5 min |
| 4 | DistilBERT + LoRA | Google Colab, GPU T4 | 20–40 min |
| 5 | Capstone | Lectura del informe final | — |

Los tiempos son aproximados y dependen del equipo, la conexión y la disponibilidad de GPU.

## 4. Módulo 1 — Pipeline base de PyTorch

El dataset Iris se carga directamente desde scikit-learn; no requiere descargas manuales.

### Windows PowerShell

```powershell
cd 01_modulo1_pipeline_pytorch
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py
deactivate
cd ..
```

### macOS o Linux

```bash
cd 01_modulo1_pipeline_pytorch
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py
deactivate
cd ..
```

Resultados esperados: `results/metrics.json` y `results/training_curves.png`.

## 5. Dataset común de los módulos 2, 3 y 4

Los módulos de NLP utilizan las mismas particiones de **AG News**:

```text
data/ag_news_train.csv
data/ag_news_test.csv
```

Seguí los enlaces e instrucciones de:

- [Datos del Módulo 2](02_modulo2_eda_nlp/data/README.md)
- [Datos del Módulo 3](03_modulo3_tfidf_baseline/data/README.md)

El test debe mantenerse aislado durante el EDA y el ajuste de parámetros. Solo se utiliza para la evaluación final.

## 6. Módulo 2 — Preprocesamiento y EDA

### Windows PowerShell

```powershell
cd 02_modulo2_eda_nlp
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/preprocess_eda.py
deactivate
cd ..
```

### macOS o Linux

```bash
cd 02_modulo2_eda_nlp
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/preprocess_eda.py
deactivate
cd ..
```

Resultados esperados: tablas de frecuencias y n-gramas en `results/`, además de visualizaciones en `reports/figures/`.

## 7. Módulo 3 — Baseline TF-IDF

Copiá `ag_news_train.csv` y `ag_news_test.csv` dentro de `03_modulo3_tfidf_baseline/data/`.

### Windows PowerShell

```powershell
cd 03_modulo3_tfidf_baseline
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train_evaluate.py
deactivate
cd ..
```

### macOS o Linux

```bash
cd 03_modulo3_tfidf_baseline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train_evaluate.py
deactivate
cd ..
```

Resultados esperados: métricas, reporte de clasificación, matriz de confusión y ejemplos mal clasificados en `results/`.

## 8. Módulo 4 — DistilBERT + LoRA

Se recomienda Google Colab porque el entrenamiento fue diseñado para una GPU NVIDIA T4.

1. Abrí [el notebook del Módulo 4](04_modulo4_distilbert_lora/Noero_Matias_LoRA_AGNews.ipynb).
2. En Colab, seleccioná **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**.
3. Ejecutá las celdas en orden desde el comienzo.
4. Si la instalación modifica paquetes, reiniciá el entorno cuando Colab lo solicite y volvé a ejecutar desde la primera celda.
5. Descargá el ZIP generado al finalizar.

La instalación del notebook elimina una versión incompatible de `torchao` y fija rangos compatibles de Transformers, Datasets, PEFT y Accelerate.

Resultados esperados: métricas de LoRA, historial de entrenamiento, matriz de confusión y comparación con el baseline.

## 9. Proyecto final

Los entregables finales están en [`05_proyecto_final_capstone/`](05_proyecto_final_capstone/):

- informe técnico consolidado;
- resumen integral del curso;
- presentación visual.

Ambos clasificadores se comparan sobre el mismo `ag_news_test.csv` de 2.000 noticias.

## 10. Solución de problemas

### PowerShell no permite activar el entorno

Ejecutá una vez en la terminal actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Luego repetí:

```powershell
.venv\Scripts\activate
```

### No se encuentra un módulo de Python

Comprobá que el entorno esté activado y reinstalá las dependencias del módulo actual:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### No se encuentra el dataset

Verificá el nombre exacto y la carpeta esperada:

```text
ag_news_train.csv
ag_news_test.csv
```

### Colab muestra conflictos de paquetes

Usá el notebook publicado en el Módulo 4 y ejecutá primero su celda de instalación. No agregues una instalación global de `pandas` ni de `torchao`.

## 11. Resultados de referencia

| Modelo | F1 weighted | Accuracy |
|---|---:|---:|
| TF-IDF + LinearSVC | 0,8925 | 0,8925 |
| DistilBERT + LoRA | 0,9070 | 0,9070 |

Pequeñas diferencias pueden aparecer por versiones, hardware o backend, aunque las semillas y configuraciones están documentadas.
