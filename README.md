# MachineLearningProject_TB
This repository contains the Machine Learning Module Project of the September 2023 The Bridge Data Science Bootcamp.

El tema de este proyecto es crear un modelo de machine learning mediante python. Para ello utilizamos un set de datos procendente de Kaggle (https://www.kaggle.com/datasets/yeoyunsianggeremie/most-popular-python-projects-on-github-2018-2023). Dicho set de datos contiene el top 100 de repostorios de python en GitHub desde diciembre de 2018. 

Los puntos a seguir fueron:

- Carga de datos y exploración inicial: se explora el set de datos en python para conocer su cualidades y características.
- EDA:  un análisis exploratorio que nos permite saber como se distribuyen y relacionan los datos.
- Preprocesamiento: se utilizaron diversos métodos de preprocesamiento de los datos para probar con diferentes modelos.
- Modelos: apoyándonos en el paso anterior, se crearon diferentes modelos con el fin de encontrar el que mejor prediga.
- Selección del modelo: Selección del modelo más eficaz.

El objetivo es predecir la cantidad de forks que van a obtener los repositorios. 

Los preprocesamientos aplicados en el modelo final fueron:
- Limpieza del dataset: se eliminaron columnas que solo aportaban ruido.
- Feature Engineering: se crearon columnas a partir de las ya existentes para mejorar la eficacia del modelo.
- Natural Language Processing: se utilizaron técnicas de NLP para estimar el sentimiento de la descripción de los repositorios.
- Tranformación: se utilizó el método LabelEncoder para transformar las columnas no númericas.
- Escalado: se escalaron los datos mediante el uso de RobustEscaler.
- Clusterización:  se utilizó un modelo KMeansMiniBacth para encontrar cluster o grupos que poder predecir por separado.

El modelo elegido fue el árbol de decisión, ya que con eficacias similares a RandomForest y Histogram-based Gradient Boosting explica los resultados de una manera mucha más sencilla. Este modelo se aplicó a los 5 clusters creados por el KMeansMiniBatch.
