#! python

from pathlib import Path
from labels import CODE_TO_CANCER, CODE_TO_COMMON_NAME, INDEX_TO_CODE
import tensorflow as tf
import os

class Predict:
    IMAGE_SIZE: tuple[int, int] = (28, 28)
    RESCALING: float = 1./255
    MODEL_DIRECTORY = Path(__file__).resolve().parent / 'saved_models'

    def __init__(self) -> None:
        model_path = self._get_lastest_model()
        self.load_model(model_path)

    """
    Returns the path to the latest model.
    """
    def _get_lastest_model(self) -> str:
        directory_walk = next(os.walk(self.MODEL_DIRECTORY))
        model_paths = [ f"{directory_walk[0]}/{x}" for x in directory_walk[1] ]
        return f"{max(model_paths)}"

    """
    Loads a model from a SavedModel directory.
    """
    def load_model(self, model_path: str) -> None:
        self.model = tf.keras.models.load_model(model_path)

    """
    Predicts the class of a given image.
    """
    def predict(self, file_path: str) -> tf.Tensor:
        x = self._preprocess_image(file_path)
        y_pred = self.model.predict([x])
        y_pred_index = tf.argmax(y_pred, axis=1).numpy()[0]
        return INDEX_TO_CODE[y_pred_index]

    """
    Preprocesses an image.
    """
    def _preprocess_image(self, file_path: str) -> tf.Tensor:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

if __name__ == "__main__":
    import sys

    # Get first argument: image path
    image_path = sys.argv[1]

    predict = Predict()
    y_code = predict.predict(image_path)
    print(f"Predicted class: {y_code}, Name: {CODE_TO_COMMON_NAME[y_code]}, Cancer?: {CODE_TO_CANCER[y_code]}")
