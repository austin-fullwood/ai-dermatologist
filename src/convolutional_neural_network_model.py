import tensorflow as tf

"""
Creates a TensorFlow convolutional neural network model.
"""
class ConvolutionalNeuralNetworkModel(tf.keras.Model):
    """
    Initializes the CNN layers.

    Args:
        num_of_classes (int): The number of classes in the dataset.
        image_size (tuple[int, int]): The image size.
        rescaling (float): The rescaling factor.
    """
    def __init__(
            self,
            num_of_classes: int,
            image_size: tuple[int, int]=(28, 28),
            rescaling: float=1./255
        ) -> None:
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Create model layers
        self.resizing = tf.keras.layers.Resizing(image_size[0], image_size[1])
        self.rescaling = tf.keras.layers.Rescaling(rescaling)
        self.random_flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")
        self.random_rotation = tf.keras.layers.RandomRotation(0.2)
        self.conv1 = tf.keras.layers.Conv2D(16, 2, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 2, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(64, 2, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_of_classes, activation="softmax")

    """
    Performs a forward pass on the model.

    Args:
        inputs (tf.Tensor): The input data.
        training (bool): Whether the model is training or not.
    Returns:
        tf.Tensor: The output data.
    """
    def call(self, inputs: tf.Tensor, training: bool=False) -> tf.Tensor:
        x = self.resizing(inputs)
        x = self.rescaling(x)
        if training:
            x = self.random_flip(x)
            x = self.random_rotation(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        return x