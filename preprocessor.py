import numpy as np
from PIL import Image, ImageEnhance

class Preprocessor:
    @staticmethod
    def normalizeImage(image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_resized = image.resize((224, 224))
        image_resized = np.array(image_resized) / 255.0
        if image_resized.ndim == 2:
            image_resized = np.stack([image_resized] * 3, axis=-1)
        image_resized = np.expand_dims(image_resized, axis=0)
        return image_resized