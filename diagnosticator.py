import tensorflow as tf

class Diagnosticator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def hasGlaucoma(self, image_resized):
        prediction = self.model.predict(image_resized)
        return 'El ojo tiene glaucoma' if prediction[0] < 0.7 else 'El ojo no tiene glaucoma'