<div align="center">

![logo](https://storage.googleapis.com/kaggle-datasets-images/54339/103727/d0f9325d9f7d113719887bc22235e5d4/dataset-cover.png?t=2018-09-19-17-09-26)

<h3>
AI Dermetologist
</h3>

Identifies skin lesions in images. Maintained by [austin fullwood](https://www.austinfullwood.com).

</div>

---

## Overview
A Convolutional Neural Network built with TensorFlow takes in an image file, preprocesses it, and outputs a list of probabilities for each of the 7 classes. The class with the highest probability is deemed the predicted guess.

The CNN is trained on the [Skin MNIST dataset](https://github.com/bundasmanu/skin_mnist) with 10015 different images of different skin lesions.

## Getting Started

Python 3.11.x is required for this project. Please download it [here](https://www.python.org/downloads/).

Download dependencies:
```
pip install -r requirements.txt
```
## Training Model

Run the training script:
```
python src/train.py
```

This will download the dataset (requires [Kaggle account](https://www.kaggle.com/docs/api#:~:text=is%20%24PYTHON_HOME/Scripts.-,Authentication,-In%20order%20to)), train a CNN, and save the model to the `saved_models` directory.

## Run Model

Run the prediction script:
```
python src/run.py <image_path>
```

This will load the lastest trained model and output a prediction.
