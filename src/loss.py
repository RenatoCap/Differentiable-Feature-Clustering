import tensorflow as tf
import keras


class CustomLoss(keras.losses.Loss):
    """
        Custom Loss for our Differential Feature Cluster Implementation with TensorFlow,
        these loss consists of a constraint on feature similiarity and constraint on spatial
        continuity

        Loss = Feature Similiarity Loss + mu * Spatial Continuity Loss
    """

    def __init__(self, mu, q, width, height):
        """ Creates a Loss function that we use in the training of our neural network

            Args:
            mu: represents the weight for balancing the two constraints.
            q: Number of clusters of our convolutional 1D Layer
            width: width shape of the image
            height: height shape of the image

        """
        super().__init__(name='custom_loss')

        self.mu = mu
        self.q = q
        self.width = width
        self.height = height

    def feat_sim_loss(self, y_true, y_pred):
        # feature similarity loss
        response_pixels = tf.reshape(y_pred, [-1, self.q])
        cluster_labels = tf.argmax(y_pred, axis=3, output_type=tf.dtypes.int64)
        cluster_labels = tf.reshape(cluster_labels, [-1, 1])
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        sim_loss = scce(cluster_labels, response_pixels)  # tf.keras.losses.SparseCategoricalCrossentropy()
        sim_loss = tf.cast(sim_loss, dtype='float32')
        return sim_loss

    def continuity_loss(self, y_true, y_pred):
        # spatial continuity loss
        img_reshape = tf.reshape(y_pred, (self.width, self.height, self.q))
        img_dx = img_reshape[:, 1:, :] - img_reshape[:, 0:-1, :]
        img_dy = img_reshape[1:, :, :] - img_reshape[0:-1, :, :]
        dx_target = tf.zeros(img_dx.shape, dtype='float32')
        dy_target = tf.zeros(img_dy.shape, dtype='float32')
        mae = tf.keras.losses.MeanAbsoluteError()
        mae_y = mae(img_dy, dy_target)
        mae_x = mae(img_dx, dx_target)

        continu_loss = mae_x + mae_y
        return continu_loss

    def call(self, y_true, y_pred):
        """Calculate feature similiarity loss and spatial continuity loss

           Args:
           y_pred: The output of our neural network.

        """
        # Return the loss that we use in the training
        return self.feat_sim_loss(y_true, y_pred) + self.mu * self.continuity_loss(y_true, y_pred)
