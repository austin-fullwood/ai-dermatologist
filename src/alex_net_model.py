import tensorflow as tf

"""
Creates a TensorFlow AlexNet model.
"""
class AlexNetModel(tf.keras.Model):
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
            image_size: tuple[int, int]=(227, 227),
            rescaling: float=1./255
        ) -> None:
        super(AlexNetModel, self).__init__()

        # Image augmetation
        self.resizing = tf.keras.layers.Resizing(image_size[0], image_size[1])
        self.rescaling = tf.keras.layers.Rescaling(rescaling)
        self.random_flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")
        self.random_rotation = tf.keras.layers.RandomRotation(0.2)

        # Layer 1
        self.conv1 = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        # Layer 2
        self.conv2 = tf.keras.layers.Conv2D(256, (5, 5), padding="same", activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        # Layer 3
        self.conv3 = tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

        # Layer 4
        self.conv4 = tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu')
        self.batch_norm4 = tf.keras.layers.BatchNormalization()

        # Layer 5
        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')
        self.pool5 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
        self.batch_norm5 = tf.keras.layers.BatchNormalization()

        # Flatten
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(num_of_classes, activation="softmax")

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
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = self.batch_norm5(x)

        x = self.flatten(x)

        x = self.dense1(x)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
        x = self.dense3(x)
        return x