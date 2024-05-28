PROYECTO DE DETECCIÓN Y RECONOCIMIENTO FACIAL
Descripción
Este proyecto implementa una aplicación de detección y reconocimiento facial utilizando Python y OpenCV. La aplicación detecta y reconoce rostros en imágenes, realizando un proceso completo desde el preprocesamiento de las imágenes hasta la evaluación de un modelo de reconocimiento facial.

Componentes del Proyecto
Preprocesamiento de Imágenes:

Cargar imágenes desde un directorio especificado.
Convertir las imágenes a escala de grises.
Aplicar ecualización del histograma para mejorar el contraste de las imágenes.
Detección de Rostros:

Utilizar el clasificador Haar Cascade (haarcascade_frontalface_default.xml) de OpenCV para detectar rostros en las imágenes preprocesadas.
Dibujar rectángulos alrededor de los rostros detectados y guardar las imágenes resultantes en un nuevo directorio.
Reconocimiento Facial:

Implementar un sistema de reconocimiento facial utilizando el algoritmo Local Binary Patterns Histograms (LBPH) de OpenCV.
Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
Entrenar el modelo LBPH con las imágenes de entrenamiento.
Evaluar la precisión del modelo utilizando las imágenes de prueba.
Requisitos
Python 3.x
OpenCV
NumPy
scikit-learn
