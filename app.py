import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageEnhance
import io
import base64
from scipy.ndimage import label, find_objects

# Cargamos el modelo segmentador.
CNN_Segmenter = load_model('ONHDetector.h5')

app = Flask(__name__)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Reducimos la exposición en un 30%
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(.7) 
    image_resized = image.resize((224, 224))
    image_resized = np.array(image_resized) / 255.0 # Rescalamos el rango de los pixeles a [0,1]
    if image_resized.ndim == 2: # En el caso de que no poseea 3 canales, ya que el modelo maneja 3 canales.
        image_resized = np.stack([image_resized] * 3, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)
    return image, image_resized, image.size

def get_square_bounding_box(binary_mask, original_size, crop_padding_percent):
    # Calculamos el porcentaje como fracción para nuestro bounding box.
    crop_padding = crop_padding_percent / 100.0
    # Etiquetamos la máscara binaria.
    labeled_mask, num_features = label(binary_mask)
    if num_features == 0:
        return None
    # De las etiquetas encontramos la más grande, donde haya más probabilidad.
    areas = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    largest_region_index = np.argmax(areas) + 1  # Encontrar la región con el área más grande
    # Obtenemos las coordenadas del bounding box de la región más grande
    slices = find_objects(labeled_mask == largest_region_index)[0]
    top_left_y, top_left_x = slices[0].start, slices[1].start
    bottom_right_y, bottom_right_x = slices[0].stop, slices[1].stop
    # Calculamos el centro y el tamaño del lado del cuadrado.
    center_x = (top_left_x + bottom_right_x) // 2
    center_y = (top_left_y + bottom_right_y) // 2
    side_length = max(bottom_right_y - top_left_y, bottom_right_x - top_left_x) # Obtenemos el lado más grande
    # Aplicamos el padding al cuadro delimitador, y hacemos que el bounding siempre sea cuadrado.
    top_left_x = max(center_x - side_length // 2 - int(side_length * crop_padding), 0)
    top_left_y = max(center_y - side_length // 2 - int(side_length * crop_padding), 0)
    bottom_right_x = min(center_x + side_length // 2 + int(side_length * crop_padding), original_size[0])
    bottom_right_y = min(center_y + side_length // 2 + int(side_length * crop_padding), original_size[1])
    return (top_left_y, top_left_x), (bottom_right_y, bottom_right_x)

def crop_to_square(image, bounding_box):
    top_left, bottom_right = bounding_box
    cropped_image = image.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))
    #cropped_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)
    return cropped_image

def pil_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')

def draw_bounding_box_on_mask(binary_mask, bounding_box):
    # Tenemos que asegurarnos que la máscara binaria sea bidimensional
    if len(binary_mask.shape) == 3:
        binary_mask = np.squeeze(binary_mask)  # Reduce dimensiones extra si existen
    # Convertir la máscara binaria en una imagen PIL
    mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
    draw = ImageDraw.Draw(mask_image)
    top_left, bottom_right = bounding_box
    draw.rectangle( # Aquí dibujamos el bounding box 
        [(top_left[1], top_left[0]), (bottom_right[1], bottom_right[0])],
        outline="red",
        width=2
    )
    return mask_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encuentra un archivo.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No hay un archivo seleccionado.'}), 400
    image_bytes = file.read()
    original_image, image_resized, original_size = preprocess_image(image_bytes) # Preprocesamos la imagen.
    crop_size_percent = float(request.form.get('crop_size_percent', 15)) # Obtenemos el porcentaje de padding del request.
    prediction = CNN_Segmenter.predict(image_resized) # El modelo segmentador crea una imagen de probabilidad
    # Transformamos la imagen de probabilidades en una máscara booleana.
    binary_mask = (prediction[0] > 0.5).astype(np.uint8)
    binary_mask_image = (binary_mask * 255).astype(np.uint8)
    binary_mask_image = np.squeeze(binary_mask_image)
    bounding_box = get_square_bounding_box(binary_mask, binary_mask.shape, crop_size_percent)
    if not bounding_box:
        return jsonify({'error': 'No se detectó una región válida.'}), 400
    # Dibujamos cuadro delimitador (bounding box) en la máscara binaria
    mask_with_box = draw_bounding_box_on_mask(binary_mask, bounding_box)
    # Escalamos las coordenadas del cuadro delimitador al tamaño de la imágen original original
    scale_x = original_size[0] / binary_mask.shape[1]
    scale_y = original_size[1] / binary_mask.shape[0]
    scaled_bounding_box = ( # Ahora reescalamos el bounding box
        (int(bounding_box[0][0] * scale_y), int(bounding_box[0][1] * scale_x)),
        (int(bounding_box[1][0] * scale_y), int(bounding_box[1][1] * scale_x))
    )
    cropped_image = crop_to_square(original_image, scaled_bounding_box) # Cortamos la imagen (Crop) con este bounding box.
    # Convertimos las imágenes a base64 para enviarlas por json como respuesta.
    mask_with_box_base64 = pil_to_base64(mask_with_box.convert("RGBA"))
    cropped_image_base64 = pil_to_base64(cropped_image)
    return jsonify({
        'prediction_image': mask_with_box_base64,
        'cropped_image': cropped_image_base64,
    })

if __name__ == '__main__':
    app.run(debug=True)
