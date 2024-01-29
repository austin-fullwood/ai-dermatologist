#! python
"""
Trains a CNN model on the Skin Cancer MNIST: HAM10000 dataset.
"""

import datetime
from pathlib import Path
import tensorflow as tf

from convolutional_neural_network_model import ConvolutionalNeuralNetworkModel
from skin_cancer_dataset import SkinCancerDataset

# Parameters
IMAGE_SIZE: tuple[int, int] = (28, 28)
BATCH_SIZE: int = 32
EPOCHS: int = 10
RESCALING: float = 1./255
MODEL_DIRECTORY = Path(__file__).resolve().parent / 'saved_models'

# Get dataset
skin_cancer_dataset = SkinCancerDataset(download=False)
dataset = skin_cancer_dataset.get_dataset()

# Split dataset into training and validation
total_size = len(list(dataset))
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Shuffle and batch datasets
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

# Create model and train it
model = ConvolutionalNeuralNetworkModel(
    num_of_classes=skin_cancer_dataset.num_of_classes(),
    image_size=IMAGE_SIZE,
    rescaling=RESCALING
)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=EPOCHS)

# Save model as SavedModel
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save(MODEL_DIRECTORY / f'{now}')

# Evaluate the model
loss, acc = model.evaluate(val_dataset, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Save model metadata to metadata file
with open(MODEL_DIRECTORY / 'metadata.txt', 'a') as f:
    f.write(f"TIME:\t\t{now}\n")
    f.write(f'MODEL:\t\t{type(model).__name__}\n')
    f.write(f"LOSS:\t\t{loss}\n")
    f.write(f"ACCURACY:\t{acc}\n")
    f.write("\n")

# TODO: Setup package manager

# TODO: Create API for model

# TODO: Create app that calls

# TODO: Play with model and parameters

# TODO: Look for bottlenecks with dataset loading
