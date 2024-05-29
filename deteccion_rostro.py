import cv2
import os

# Directorio de imágenes con rostros procesados
output_dir = 'procesed_faces'

# Cargar el clasificador pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear si no existe directorio de imágenes con rostros detectados
detected_faces_dir = 'detected_faces'
os.makedirs(detected_faces_dir, exist_ok=True)

# Procesar cada imagen en el directorio de imágenes procesadas
for filename in os.listdir(output_dir):
    img_path = os.path.join(output_dir, filename)
    img = cv2.imread(img_path)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3, minSize=(30,30))
    
    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    
    # Guardar la imagen con rostros detectados
    detected_path = os.path.join(detected_faces_dir, filename)
    cv2.imwrite(detected_path, img)
