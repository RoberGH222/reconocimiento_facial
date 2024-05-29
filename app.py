from flask import Flask, render_template, request, redirect, url_for
import cv2
import os

app = Flask(__name__)

# Cargar el clasificador Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Cargar el modelo entrenado LBPH
model_path = 'model/trained_model.yml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Fijo tamaño de las imágenes
fixed_size = (200, 200)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads/', filename)
        file.save(file_path)
        
        # Leer la imagen
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error al leer la imagen: {file_path}")
            return redirect(request.url)
        
        img_resized = cv2.resize(img, fixed_size)
        
        # Aplicar ecualización del histograma
        img_equalized = cv2.equalizeHist(img_resized)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            img_equalized, 
            scaleFactor=1.05,  # Ajuste del factor de escala
            minNeighbors=3,    # Ajuste del número mínimo de vecinos
            minSize=(30, 30),  # Tamaño mínimo del rostro
            maxSize=(250,250)  # Tamaño máximo del rostro
        )
        
        print(f"Faces detected: {len(faces)}")
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imwrite(file_path, img_resized)  # Guardar la imagen con detecciones
        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('uploaded.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
