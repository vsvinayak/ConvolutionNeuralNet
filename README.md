# ConvolutionNeuralNet
Implementation of a CNN. The architecture does not include training modules. The weights should be pre-trained.

Weights are extracted from a json file of pre-trained weight matrices.
The weight dictionary is of the form

'Wl i j' for convolutional layer weights, where

l - overall layer number
i - ith feature in the next layer
j - jth feature in the previous layer

'Wl' for mlp weights

l - overall layer number

For usage, see the test.py file

