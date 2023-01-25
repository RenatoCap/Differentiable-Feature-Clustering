import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


def load_image():
    """
        Load one image from cassava datasets
    """

    ds = tfds.load('cassava', split='train')

    for data in ds.take(1):
        img = data['image']

    return img


def preprocessing(image):
    """
        Normalize the data between 0 an 1, then reshape our tensor

        Arg:
        image: image that we want to preprocess

    """

    data = tf.cast(image, dtype='float32') / 255.0
    data = tf.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
    width = data.shape[1]
    height = data.shape[2]

    return data, width, height


def inference(model, image):
    """image segmented by our neural network

       Args:
       model: Difference Feature Clustering model
       image: image that we want to segmented

    """
    c_n = model(image)
    c_n = tf.argmax(c_n, axis=3)
    c_n = tf.reshape(c_n, (c_n.shape[1], c_n.shape[2]))

    return c_n


def plot_pictures(predict_image, real_image):
    """Plot the two images the segmented image and the real one

       Args:
       predict_image: Segmented Image
       real_image: real image of our dataset

    """
    plt.figure(figsize=(25, 15))
    plt.subplot(121)
    plt.imshow(real_image)
    plt.title('Image real')
    plt.subplot(122)
    plt.imshow(predict_image)
    plt.title('Image Segmented')
    plt.show()
