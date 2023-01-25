import tensorflow as tf

from loss import Loss
from utils import plot_pictures, inference, load_image, preprocessing

if __name__ == "__main__":
    image = load_image()
    data, width, height = preprocessing(image)

    # Parameters of our neural network
    p = 100
    q = 100
    M = 3

    # Construction of our neural network
    inputs = tf.keras.Input(shape=(width, height, 3))

    # Feature Extraction
    x = tf.keras.layers.Conv2D(filters=p, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    for i in range(M - 1):
        x = tf.keras.layers.Conv2D(filters=p, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

    # Response Map
    x = tf.keras.layers.Conv2D(filters=q, kernel_size=1)(x)
    output = tf.keras.layers.BatchNormalization()(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    # View the structure of our model
    model.summary()

    # Optimizer and Loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    loss = Loss(mu=0.5, q=q, width=width, height=height).losses

    model.compile(optimizer=optimizer, loss=loss)

    # Train our model
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
    model.fit(data, data, epochs=500, callbacks=[earlystopping])

    # Predict image
    image_segmented = inference(model, data)

    # Plot our result and real image
    plot_pictures(predict_image=image_segmented, real_image=image)
