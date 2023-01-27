import tensorflow as tf

from loss import CustomLoss
from utils import plot_pictures, inference, load_image, preprocessing
from diff_feature_model import DiffFeatureModel

if __name__ == "__main__":
    image = load_image()
    data, width, height = preprocessing(image)

    # Parameters of our neural network
    p = 100
    q = 100
    M = 3

    #Create a model 
    model = DiffFeatureModel(img_shape= [width, height], p=p, q=q, m=M)

    # View the structure of our model
    model.summary()

    # Optimizer and Loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    loss = CustomLoss(mu=0.5, q=q, width=width, height=height)

    model.compile(optimizer=optimizer, loss=loss)

    # Train our model
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
    model.fit(data, data, epochs=500, callbacks=[earlystopping])

    # Predict image
    image_segmented = inference(model, data)

    # Plot our result and real image
    plot_pictures(predict_image=image_segmented, real_image=image)
