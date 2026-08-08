# Datos

Este proyecto utiliza la muestra de **AG News** provista por la cátedra:

- [Descargar ag_news_train.csv](https://drive.google.com/file/d/1TSlFi4d7Nu4K-FW1QFaR3MEe2cahfxxj/view)
- [Descargar ag_news_test.csv](https://drive.google.com/file/d/18hrrgrXT6469bdOwSRhSqQvtKdcVPpYq/view)

Guardá ambos archivos con esta estructura:

```text
data/
├── ag_news_train.csv
└── ag_news_test.csv
```

Los CSV no se versionan en GitHub. El vectorizador y el modelo se ajustan
exclusivamente con `ag_news_train.csv`; `ag_news_test.csv` se utiliza una sola
vez para la evaluación final.

