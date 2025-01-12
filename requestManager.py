class RequestManager:
    def __init__(self, diagnosticator, preprocessor, imageValidator):
        self.diagnosticator = diagnosticator
        self.preprocessor = preprocessor
        self.imageValidator = imageValidator

    def diagnosticate(self, image_path):
        image_resized = self.preprocessor.normalizeImage(image_path)

        if not self.imageValidator.isValidImage(image_resized):
            self.notificateError()
            return "invalid_fundus"

        return self.diagnosticator.hasGlaucoma(image_resized)

    @staticmethod
    def notificateError():
        print("Error: La imagen no es un fondo de ojo válido.")