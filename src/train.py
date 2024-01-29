#! python

import datetime
from pathlib import Path
import tensorflow as tf

from convolutional_neural_network_model import ConvolutionalNeuralNetworkModel
from skin_cancer_dataset import SkinCancerDataset

"""
Trains a CNN model on the Skin Cancer MNIST: HAM10000 dataset.
"""
class Train:
    # Parameters
    IMAGE_SIZE: tuple[int, int] = (28, 28)
    RESCALING: float = 1./255
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    MODEL_DIRECTORY = Path(__file__).resolve().parent / 'saved_models'

    """
    Initializes the dataset and the model.
    """
    def __init__(self) -> None:
        self.dataset = SkinCancerDataset(download=True)
        self.reset()

    """
    Resets the dataset and the model.
    """
    def reset(self) -> None:
        self._set_dataset()
        self._set_model()

    """
    Set the dataset.
    """
    def _set_dataset(self) -> None:
        ds = self.dataset.get_dataset()
        # Split dataset into training and validation
        total_size = len(list(ds))
        train_size = int(0.8 * total_size)

        train_dataset = ds.take(train_size)
        val_dataset = ds.skip(train_size)

        # Shuffle and batch datasets
        self.train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.BATCH_SIZE)
        self.val_dataset = val_dataset.batch(self.BATCH_SIZE)

    """
    Sets the model.
    """
    def _set_model(self) -> None:
        self.model = ConvolutionalNeuralNetworkModel(
            num_of_classes=self.dataset.num_of_classes(),
            image_size=self.IMAGE_SIZE,
            rescaling=self.RESCALING
        )
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    """
    Trains the model.
    """
    def train(self) -> None:
        self.model.fit(self.train_dataset, epochs=self.EPOCHS)

    """
    Evaluates the model.

    Returns
        float: The loss of the model.
        float: The accuracy of the model.
    """
    def evaluate(self) -> (float, float):
        return self.model.evaluate(self.val_dataset, verbose=2)

    """
    Saves the model.

    Args:
        model (ConvolutionalNeuralNetworkModel): The model to save.
    Returns:
        str: The path to the saved model.
    """
    def save(self) -> str:
        # Save model as SavedModel
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = self.MODEL_DIRECTORY / f'{now}'
        self.model.save(file_path)

        # Save model metadata to metadata file
        with open(self.MODEL_DIRECTORY / 'metadata.txt', 'a') as f:
            f.write(f"TIME:\t\t{now}\n")
            f.write(f'MODEL:\t\t{type(self.model).__name__}\n')
            f.write(f"LOSS:\t\t{loss}\n")
            f.write(f"ACCURACY:\t{acc}\n")
            f.write("\n")

        return file_path

if __name__ == "__main__":
    train = Train()
    train.train()

    loss, acc = train.evaluate()
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    path = train.save()
    print(f"Model saved to {path}")


# TODO: Create API for model

# TODO: Create app that calls
