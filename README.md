# Differentiable Feature Clustering
TensorFlow implementation of Differential Feature Clustering, based on the paper [Unsupervised Learning of Image Segmentation
Based on Differentiable Feature Clustering](https://arxiv.org/abs/2007.09990)

## QuickStart 
Follow the instructions below to clone this repositoy in your local Machine.
```
git clone https://github.com/HimblerCap/Differentiable-Feature-Clustering.git
cd Differentiable-Feature-Clustering

python -m venv venv
source ./venv/Scripts/activate
pip install -r requirements.txt

```

## Model Arquitecture
The architecture of the model is made up of a feature extractor that contains 3 convolutional layers with filters equal to 100 accompanied by the ReLu activation function and a BatchNormalization layer. The final structure of the neural network is made up of a convolutional layer with a one-dimensional kernel and a number of filters equal to 100, culminating in a BatchNormalization layer.

![Model Arquitecture](https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/Model%20Arquitecture.png?raw=true)

## Experiments 

