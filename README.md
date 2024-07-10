# Neural Network Library + Arabic MNIST

This project is intended for educational purposes. This project implements a sequential neural network library from scratch
using NumPy. There are several options for layers, optimizers, activation functions, loss functions, and initializers.

## Layers
- 2D Convolution (with filter size and padding options)
- 2D Max Pooling (with filer size, strides options)
- Dropouts
- Activation
- Fully Connected

## Optimizers
- Adam
- RMS Prop
- Adagrad
- Gradient Descent with Momentum

## Activation Functions
- Sigmoid
- Softmax
- ReLU
- Leaky ReLU
- Tanh
- Softplus

## Loss Functions
- Hinge
- Cross Entropy
- Mean Squared Error (MSE)

## Initializers
- Glorot (Xavier) Normal
- He (Kaiming) Normal
- Lecun Normal
- Zeros
- Ones

The backpropagation procedure was tested using gradient checking (numerical differentiation)

This library was tested on Arabic MNIST dataset and got an accuracy of 85%.
