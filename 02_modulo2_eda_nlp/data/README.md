# Datos

Este proyecto utiliza la muestra de **AG News** provista por la cátedra:

- [Descargar ag_news_train.csv](https://drive.google.com/file/d/1TSlFi4d7Nu4K-FW1QFaR3MEe2cahfxxj/view)
- [Descargar ag_news_test.csv](https://drive.google.com/file/d/18hrrgrXT6469bdOwSRhSqQvtKdcVPpYq/view)
- [Script original de carga](https://drive.google.com/file/d/139M3L64c-U8pGiGZ4PmqA7TeyPszwWwV/view)
- [README de la cátedra](https://drive.google.com/file/d/15sTJ0vAenCyk7HHYM0EBhUQkIsORAugz/view)

Guardá los CSV con esta estructura:

```text
data/
├── ag_news_train.csv
└── ag_news_test.csv
```

Los archivos del dataset no se versionan en GitHub. Esto evita duplicar el
material provisto por la cátedra y mantiene separados los datos del código.

El EDA usa solamente `ag_news_train.csv`. El archivo de test se reserva para la
evaluación final y nunca se utiliza para ajustar el pipeline ni los modelos.

