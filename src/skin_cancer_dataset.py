import os
from pathlib import Path
import shutil
import tensorflow as tf
import opendatasets as od
import pandas as pd

"""
Downloads, processes, and provides the Skin Cancer MNIST: HAM10000 dataset.
"""
class SkinCancerDataset:
    DATASET_DIRECTORY = Path(__file__).resolve().parent
    DATASET_PATH = DATASET_DIRECTORY / "skin-cancer-mnist-ham10000"
    KAGGLE_URL = "https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000"

    """
    Downloads the dataset and creates a hash table for image_id to class label.

    Args:
        download (bool): Whether to download the dataset if it doesn't exist.
    """
    def __init__(self, download: bool=False) -> None:
        # Download dataset if it doesn't exist
        if download:
            self._download()

        # Get dataset metadata
        df = pd.read_csv(
            self.DATASET_DIRECTORY /
            'skin-cancer-mnist-ham10000/HAM10000_metadata.csv'
        )

        # Set class labels
        self.class_labels = {
            class_name: index
            for index, class_name in enumerate(sorted(df['dx'].unique()))
        }

        # Create hash table for image_id to class label
        class_label_indexed_by_image_id = {
            row['image_id']: self.class_labels[row['dx']]
            for _, row in df.iterrows()
        }
        self.tf_class_label_indexed_by_image_id = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(list(class_label_indexed_by_image_id.keys())),
                values=tf.constant(list(class_label_indexed_by_image_id.values())),
            ),
            default_value=tf.constant(-1)
        )

    """
    Gives the hash that maps class labels to their indices.

    Returns:
        dict[str, int]: The hash that maps class labels to their indices.
    """
    def class_labels(self) -> dict[str, int]:
        return self.class_labels

    """
    Gives the number of classes in the dataset.

    Returns:
        int: The number of classes in the dataset.
    """
    def num_of_classes(self) -> int:
        return len(self.class_labels)

    """
    Creates and gives a TensorFlow dataset containing the Skin Cancer MNIST
    image paths and a mapping method.

    Returns:
        tf.data.Dataset: Skin Cancer MNIST dataset.
    """
    def get_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.list_files(
            str(self.DATASET_PATH / '*/*'),
            shuffle=False
        )
        dataset = dataset.map(
            lambda x: self._process_path(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset

    """
    Processes the image file path and gives the image and its class label.

    Args:
        file_path (str): The image file path.
    Returns:
        (tf.Tensor, tf.Tensor): The image and its class label.
    """
    def _process_path(self, file_path: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # Get the file stem
        parts = tf.strings.split(file_path, os.path.sep)[-1]
        stem = tf.strings.split(parts, '.')[0]

        label = tf.cast(self.tf_class_label_indexed_by_image_id.lookup(stem), tf.int32)

        # Load the image file
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)

        return image, label

    """
    Downloads the dataset from Kaggle and extracts it.
    """
    def _download(self) -> None:
        if os.path.exists(self.DATASET_PATH):
            print("Dataset already downloaded. Redownload? (y/n)")
            if input() != "y":
                return()
            shutil.rmtree(self.DATASET_PATH)

        print("--- Downloading dataset ---")
        # TODO: Remove 3rd party dependency
        od.download(self.KAGGLE_URL, self.DATASET_DIRECTORY)
        print("--- Dataset downloaded ---")
