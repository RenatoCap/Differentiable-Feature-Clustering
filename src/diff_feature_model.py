import keras
import tensorflow as tf


class DiffFeatureModel(keras.Model):
    """Custom Keras class for our Diff Feature Clustering Implementation
    on Tensorflow 2.7.0"""

    def __init__(self, img_shape, p, q, m):
        """
        This method should initialize the layers used by the model, for later use on the call method.
        Args:
            - img_shape: Shape of the input image
            - p: number of filters used on each conv laayer
            - q: max number of labels that the model can segment
            - M: number of convolutional layers for the feature extractor
        """

        super(DiffFeatureModel, self).__init__()
        self.shape = img_shape
        self.p = p
        self.q = q
        self.M = m
        self.conv1 = keras.layers.Conv2D(filters=self.p, kernel_size=3, padding='same')
        self.relu1 = keras.layers.Activation('relu')
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.conv_list = [keras.layers.Conv2D(filters=self.p, kernel_size=3, padding='same') for i in range(self.M - 1)]
        self.relu_list = [keras.layers.Activation('relu') for i in range(self.M - 1)]
        self.batchnorm_list = [keras.layers.BatchNormalization() for i in range(self.M - 1)]
        self.conv_response = keras.layers.Conv2D(filters=self.q, kernel_size=1)
        self.batchnorm_response = keras.layers.BatchNormalization()
        self.build(input_shape=(1, self.shape[0], self.shape[1], 3))

    def call(self, inputs, **kwargs):
        """This method implements the forward pass of the algorithm.
        1st step: Conv2D, ReLU, BatchNormalization (specifying input dim)
        2nd step: Conv2D, RelU, BatchNormalization ((nConv - 1) times)
        3rd step: 1X1 Conv2D
        4th step: BatchNorm
        5th step: Argmax
        """
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        for i in range(self.M - 1):
            x = self.conv_list[i](x)
            x = self.relu_list[i](x)
            x = self.batchnorm_list[i](x)

        x = self.conv_response(x)
        return self.batchnorm_response(x)

    def train_step(self, data):
        input_img = data

        with tf.GradientTape() as tape:
            output = self(input_img, training=True)
            loss = self.compiled_loss(output, output)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(target=loss, sources=trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(output, output)
        return {m.name: m.result() for m in self.metrics}
