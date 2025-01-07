from flask import Flask, render_template, request, jsonify
from diagnosticator import Diagnosticator
from preprocessor import Preprocessor
from imageValidator import ImageValidator
from requestManager import RequestManager

# Configuración
app = Flask(__name__)

validator_model_path = 'vgg16_fondo_del_ojo.keras'
diagnosticator_model_path = 'ClasificadoraSinRecorte2.keras'

# Inicializamos las clases
preprocessor = Preprocessor()
validator = ImageValidator(validator_model_path)
diagnosticator = Diagnosticator(diagnosticator_model_path)
manager = RequestManager(diagnosticator, preprocessor, validator)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnosticate', methods=['POST'])
def diagnosticate():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró un archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó un archivo'}), 400

    # Procesar la imagen y diagnosticar
    try:
        result = manager.diagnosticate(file)
        
        if result == "invalid_fundus":
            return jsonify({'error': 'La imagen no es válida como fondo del ojo'}), 400
        
        # Si la imagen es válida, retornamos el diagnóstico
        if result == 'El ojo tiene glaucoma':
            return jsonify({'result': 'El ojo tiene glaucoma'}), 200
        else:
            return jsonify({'result': 'El ojo no tiene glaucoma'}), 200

    except Exception as e:
        # Captura cualquier excepción inesperada
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)