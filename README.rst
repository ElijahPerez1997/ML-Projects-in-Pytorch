++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
README file for ML-Projects-in-Pytorch
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This repository belongs to Elijah Perez. This repository contains two machine learing projects in Python (Pytorch). Note that both of these neural networks run off of the CPU and thus do dont require a GPU to run. This does have a noticable affect on runtime, and both nets would run much faster on a GPU. 



FCNeuralNet.py:

This file implements a fully connected neural network. This neural net uses the MNIST data set of hand drawn digits (hand-drawn numbers from 0 to 9). It then learns from this data set. After learning, it can predict the true value of a drawn digit with high accuracy.



ConvNeuralNet.py:

This file implements a convolutional neural network (CNN). This CNN uses the Kaggle Cats and Dogs Datase found on Microsoft's website. To run ConvNeuralNet.py, you first need to extract the zip folder found at the given Microsoft URL (https://www.microsoft.com/en-us/download/details.aspx?id=54765) into the same working directory as the ConvNeuralNet.py file. This will extract a folder called PetImages that contains images of cats and dogs. The CNN learns from these images. After learning, the CNN can differentiate between cat and dog images.
