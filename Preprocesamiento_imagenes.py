import cv2
import os

# Directorios
input_dir = 'faces'
output_dir = 'procesed_faces'


# Fijo tamaño de las imágenes
fixed_size = (200, 200)

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Cargar y procesar cada imagen en el directorio de entrada
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        # Convertir a escala de grises
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Redimensionamos la imagen
        img_resized = cv2.resize(gray_img, fixed_size)
        
        # Aplicar histogram equalization
        equalized_img = cv2.equalizeHist(img_resized)
        
        # Guardar la imagen procesada
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, equalized_img)
