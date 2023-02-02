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

## Model Architecture
The architecture of the model is made up of a feature extractor that contains 3 convolutional layers with filters equal to 100 accompanied by the ReLu activation function and a BatchNormalization layer. The final structure of the neural network is made up of a convolutional layer with a one-dimensional kernel and a number of filters equal to 100, culminating in a BatchNormalization layer.

![Model Arquitecture](https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/Model%20Arquitecture.png?raw=true)

## Experiments 

<div style="display: table; clear: both; content: """>
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/real.png" width="30%" height="30%" style="vertical-align: text-bottom;">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_01.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_009.png" width="30%" height="30%">
</div>

<div style="display: table; clear: both; content: """>
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_008.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_007.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_006.png" width="30%" height="30%">
</div>

<div style="display: table; clear: both; content: """>
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_005.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/Adam_004.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/SGD_01_09.png" width="30%" height="30%">
</div>

<div style="display: table; clear: both; content: """>
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/p_100_q_128.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/p_100_q_150.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/p_100_q_32.png" width="30%" height="30%">
</div>

<div style="display: table; clear: both; content: """>
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/p_100_q_64.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/u_10.png" width="30%" height="30%">
  <img src="https://github.com/HimblerCap/Differentiable-Feature-Clustering/blob/master/img/experiment_img/u_10_p_150.png" width="30%" height="30%">
</div>

