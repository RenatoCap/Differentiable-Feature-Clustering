import keras
import argparse
import matplotlib.pyplot as plt

from utils import load_image, preprocessing


def view_image_per_layer(model, data):
    """ Plot the output of each layer except Activation layer

    Args:
        model: the model that we want to see the outputs
        data: the image that we want to predict

    Return:
        9 images plotted in a 3x3 matrix
    """

    # List with all the layers
    layers = model.layers
    layers = [layer for layer in layers if layer.name[:-3] != 'activation']

    # Creat subplots and figsize
    fig, axs = plt.subplots(3,3, figsize=(5,5))

    # Plot real image
    axs[0, 0].imshow(data[0, :, :, :])
    axs[0, 0].set_title('Original Image')

    # Plot one of the outputs of our neural network in each layer
    for i, ax in enumerate(axs.flat[1:]):
        l = layers[i](data) if i==0 else layers[i](l)
        ax.imshow(l[0, :, :, 0])
        ax.set_title(layers[i].name[:-3])

    # Add padding between images     
    fig.tight_layout(pad=2)

    # Show or plots
    plt.show()


def view_all_feature(model, data):
    """See all the characteristics of the output of a layer of our neural network
    
    Args: 
        model: the model that we want to see the outputs
        layer: is a number, indicate what output layer want to see
        data: the image that we want to predict

    Return: 
        100 images plotted in a 5x20 matrix
    """

    layers = model.layers
    outputs = []

    for idx in range(len(layers)):
        l = layers[idx](data) if idx==0 else layers[idx](l)
        outputs.append(l)

    fig, axs = plt.subplots(5, 20, figsize=(25, 25))

    for i, ax in enumerate(axs.flat[:]):
        ax.imshow(outputs[-1][0, :, :, i-1])
        ax.set_title(f'Feature {i}')

    fig.tight_layout(pad=1)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-one", type=bool, default=True, help='Graph image per layers', required=False)
    parser.add_argument("-all", type=bool, default=False, help='Graph image per layers', required=False)

    args = parser.parse_args()

    #Load image
    image = load_image()

    #Preprocess the image
    data, width, height = preprocessing(image)

    #Load Model
    model = keras.models.load_model('../models/model_01')
    
    if(args.one):
        view_image_per_layer(model, data=data)
    elif(args.all):
        view_all_feature(model, data=data)