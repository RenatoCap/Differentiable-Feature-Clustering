import tensorflow as tf 
import keras

class DiffFeatureModel(keras.Model):
    """Custom Keras class for our Diff Feature Clustering Implementation
    on Tensorflow 2.7.0"""

    def __init__(self, img_shape, p, q, M):
        """This method should initialize the layers used by the model, for later use on the call method.
        Args:
            - img_shape: Shape of the input image
            - M: number of convolutional layers for the feature extractor
            - p: number of filters used on each conv laayer
            - q: number of """
        pass
    
    def call(self, inputs, training=None, mask=None):
        """This method implements the forward pass of the algorithm.
        1st step: Conv2D, ReLU, BatchNormalization (specifying input dim)
        2nd step: Conv2D, RelU, BatchNormalization ((nConv - 1) times)
        3rd step: 1X1 Conv2D
        4th step: BatchNorm
        5th step: Argmax
        """
        pass
    
    def train_step(self, data):
        # gradienttape
        # forward
        # loss
        # apply gradients
        pass