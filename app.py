import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageEnhance
import io
import base64
from scipy.ndimage import label, center_of_mass

# Cargar el modelo
model = load_model('ONHDetector.h5')

app = Flask(__name__)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
         
    # Reducir exposición (brillo) en un 80%
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(.7)  # 20% del brillo original (-80%)

    image_resized = image.resize((224, 224))
    image_resized = np.array(image_resized) / 255.0
    if image_resized.ndim == 2: # En el caso de que no poseea 3 canales, ya que el modelo maneja 3 canales.
        image_resized = np.stack([image_resized] * 3, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)
    return image, image_resized, image.size

def get_threshold_from_top_percent(prediction, top_percent=5):
    flattened_prediction = prediction.flatten()
    sorted_predictions = np.sort(flattened_prediction)[::-1]
    top_index = int(len(sorted_predictions) * (top_percent / 100))
    top_predictions = sorted_predictions[:top_index]
    threshold = np.min(top_predictions)
    return threshold

def get_center_of_binary_mask(cropped_mask):
    labeled_mask, num_features = label(cropped_mask)
    if num_features == 0:
        return None, None, None
    areas = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    largest_region_index = np.argmax(areas) + 1
    center_of_mass_coords = center_of_mass(cropped_mask, labeled_mask, largest_region_index)
    center_y, center_x = center_of_mass_coords
    center_y, center_x = int(center_y), int(center_x)
    binary_mask = np.where(labeled_mask == largest_region_index, 1, 0).astype(np.uint8)
    return binary_mask, center_x, center_y

def crop_image(image, center_x, center_y, size_percent, original_size):
    max_side = max(original_size) 
    crop_size = int(max_side * size_percent / 100) 
    left = max(0, center_x - crop_size // 2)
    top = max(0, center_y - crop_size // 2)
    right = min(image.width, center_x + crop_size // 2)
    bottom = min(image.height, center_y + crop_size // 2)

    if left >= right or top >= bottom:
        return image  
    cropped_image = image.crop((left, top, right, bottom))  # Recorta la imagen.
    return cropped_image

def pil_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    crop_size_percent = float(request.form.get('crop_size_percent', 20))  # Tamaño predeterminado 20% si no se proporciona

    image_bytes = file.read()
    original_image, image_resized, original_size = preprocess_image(image_bytes)

    prediction = model.predict(image_resized)

    max_prediction_value = np.max(prediction[0])
    normalized_prediction = prediction[0] / max_prediction_value
    binary_mask = (normalized_prediction > 0.5).astype(np.uint8)

    binary_mask_image = (binary_mask * 255).astype(np.uint8)
    binary_mask_image = np.squeeze(binary_mask_image)

    prediction_image_non_binary = (prediction[0] * 255).astype(np.uint8)  # Convertir a imagen no binaria
    prediction_image_non_binary = np.squeeze(prediction_image_non_binary)

    mask_image_non_binary = Image.fromarray(prediction_image_non_binary).convert("RGBA")
    mask_image_non_binary = ImageEnhance.Brightness(mask_image_non_binary).enhance(0.5)  # Ajustar la máscara para alpha

    image_resized_rgba = Image.fromarray((image_resized[0] * 255).astype(np.uint8)).convert("RGBA")
    image_resized_rgba = ImageEnhance.Brightness(image_resized_rgba).enhance(0.5)

    combined_image = Image.alpha_composite(image_resized_rgba, mask_image_non_binary)

    binary_mask, center_x, center_y = get_center_of_binary_mask(binary_mask_image)

    if binary_mask is None:
        return jsonify({
            'error': 'Probablemente no es una imagen del fondo del ojo o no se detectó el disco óptico.'
        }), 400

    # Dibujamos el centro relativo de la imagen original si es que existe 
    if center_x is not None and center_y is not None:
        draw = ImageDraw.Draw(combined_image)
        center_radius = 5
        draw.ellipse([center_x - center_radius, center_y - center_radius,
                      center_x + center_radius, center_y + center_radius],
                     fill="blue", outline="blue")  # Centro en azul (sin escalar)

        scale_x = original_size[0] / 224
        scale_y = original_size[1] / 224
        center_x_original = int(center_x * scale_x)
        center_y_original = int(center_y * scale_y)

    # Recortamos la imagen alrededor del centro usando el porcentaje del tamaño de la imagen original
    cropped_image = crop_image(original_image, center_x_original, center_y_original, crop_size_percent, original_size)

    prediction_image_base64 = pil_to_base64(combined_image)
    cropped_image_base64 = pil_to_base64(cropped_image)
    binary_mask_base64 = pil_to_base64(Image.fromarray(binary_mask_image).convert("RGBA"))
    
    return jsonify({
        'prediction_image': prediction_image_base64,
        'cropped_image': cropped_image_base64,
        'binary_mask_image': binary_mask_base64  
    })

if __name__ == '__main__':
    app.run(debug=True)
