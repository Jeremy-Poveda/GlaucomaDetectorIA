import tensorflow as tf

class ImageValidator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def isValidImage(self, image_resized):
        prediction = self.model.predict(image_resized)
        return prediction[0] < 0.5  # True si es una imagen válida