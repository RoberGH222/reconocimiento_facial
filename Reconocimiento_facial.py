import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np

# Directorio con las imágenes de rostros detectados
detected_faces_dir = 'detected_faces'

# Directorio con las imágenes de rostros recortadas
faces_dir = 'cut_faces'
os.makedirs(faces_dir, exist_ok=True)

# Cargar el clasificador pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in os.listdir(detected_faces_dir):
    img_path = os.path.join(detected_faces_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_path = os.path.join(faces_dir, filename)
        cv2.imwrite(face_path, face)

# Cargar las imágenes de caras y sus etiquetas
X, y = [], []
label_dict = {}

fixed_size = (100, 100)  # Tamaño fijo para las imágenes

for i, filename in enumerate(os.listdir(faces_dir)):
    img_path = os.path.join(faces_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        continue

    # Redimensionar la imagen al tamaño fijo
    img_resized = cv2.resize(img, fixed_size)
    
    label = filename.split('_')[0]  # Asumiendo que el nombre del archivo contiene la etiqueta
    if label not in label_dict:
        label_dict[label] = len(label_dict)
    
    X.append(img_resized)
    y.append(label_dict[label])

X = np.array(X)
y = np.array(y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenar el modelo
recognizer.train(X_train, y_train)

# Evaluar el modelo
correct = 0
for i, face in enumerate(X_test):
    label, confidence = recognizer.predict(face)
    if label == y_test[i]:
        correct += 1

accuracy = correct / len(y_test)
print(f'Precisión del reconocimiento facial: {accuracy:.2f}')