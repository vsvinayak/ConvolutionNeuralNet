# ConvolutionNeuralNet

Implementation of a CNN. The architecture does not include training modules. The weights should be pre-trained.

Weights are extracted from a json file of pre-trained weight matrices.

The weight dictionary is of the form

**Wl i j**  - for convolutional layer weights, where

l - overall layer number
i - ith feature in the next layer
j - jth feature in the previous layer and

**Wl** for mlp weights

l - overall layer number


For demo

1. face_test.py - contains a gender classification demo on webcam by faces   detected by opencv's face detection
2. face_det.py - gender classification demo on webcam by face detection tracking using Optical Flow
3.  test.py - contains gender detection demo on still images
For usage, see the test.py file


