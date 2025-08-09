# C++ Neural Network with Eigen from scratch
##  Libraries used
- MNIST reader (MNIST data included)
- Eigen
- Standard Library (C++)
## Description
I made this simple neural network in C++ by utilizing Eigen's fast matrix & vector calculation.
I chose to create a neural network from scratch in C++ so that I can better understand how these work and can learn actual performance & accuracy strategies.
By making this project I gained lots of experience in both C++ (in which I've never had a real project in before) and neural networks .
Currently only a handful of activation functions are supported, but im planning on adding more for the future.
Because Eigen uses CPU mainly, GPU is not yet supported even though GPU would offer great performance benefits with multithreading and multiprocessing.
## Usage:
### Creating the network
The input layer is implicit and is not represented by a layer.
After all hidden layers/ouput layer have been added, the setup function __has__ to be called.

``` Cpp
// ...

#include "neural_network.hpp"

int main() {
  // Defines how many input vectors get processed at the same time
  const int batchSize = 10;
  const int inputLength = 784;

  // Creating the nn
  NN::Network nn(batchSize);

  // Adding layers with neurons & activation function
  nn.addLayer(16, LEAKY_RELU);
  nn.addLayer(16, LEAKY_RELU);
  nn.addLayer(10, LEAKY_RELU);

  // Mandatory setup function that has to be called after adding all layers
  nn.setup(inputLength);

  // Training... (see 'mnist_training' script for training example)

  return 0;
}
```
### Forwardpass + backwardpass
To utilize the forward- and backwardpass you just have to call their function.
Actual training example is located in the 'mnist_training' script. The following code is simplified to showcase the args.
Inputs and targets is a Matrix because of batching. If batching would be set to one the matrix should only contain one input vector.
``` Cpp 
// ...

#include "neural_network.hpp"

int main() {
  // Defines how many input vectors get processed at the same time
  const int batchSize = 10;
  const int inputLength = 784;

  // Creating the nn
  NN::Network nn(batchSize);

  // Adding layers with neurons & activation function
  nn.addLayer(16, LEAKY_RELU);
  nn.addLayer(10, LEAKY_RELU);

  // Mandatory setup function that has to be called after adding all layers
  nn.setup(inputLength);

  MatrixXf inputs = ...; // Input matrix
  MatrixXf target = ...; // Target matrix as the one hot encoded label values
  const float learning_rate = 0.009f;

  // Forwardpass
  nn.forwardpass(inputs) // Only takes inputs as args

  // Backwardpass
  nn.backwardpass(learning_rate, target, input)

  return 0;
}
```

## Example output data on MNIST
### Network
``` Cpp

Input layer size: 784
Hidden layer 1 size: 32
Hidden layer 2 size: 32
Output layer size: 10
Total weights: 25472
```


### Output
``` Cpp
Program started
Using leaky relu weight distribution for layer 0
Using leaky relu weight distribution for layer 1
Using leaky relu weight distribution for layer 2
Using leaky relu bias distribution for layer 0
Using leaky relu bias distribution for layer 1
Using leaky relu bias distribution for layer 2
Epoch 1/5 | Loss: 0.465728 | Accuracy: 64.64% | Time taken: 30 seconds
Epoch 2/5 | Loss: 0.24 | Accuracy: 86.27% | Time taken: 30 seconds
Epoch 3/5 | Loss: 0.15 | Accuracy: 92.80% | Time taken: 31 seconds
Epoch 4/5 | Loss: 0.14 | Accuracy: 93.84% | Time taken: 30 seconds
Epoch 5/5 | Loss: 0.13 | Accuracy: 94.29% | Time taken: 29 seconds
Test Accuracy: 94.22%
Training finished
```
## Problems
Because Eigen doesn't support GPU the neural network struggles with performance, even though batching is implemented and efficent matrix calcuations are being used.
That's why im planing to switch to CUDA for GPU support.
## Summary
Altough the performance is lagging behind, I could get the accuracy up to 94 to 96% accuracy consistently. The main purpose of this neural network was so that I could learn how to implement a neural network completely from scratch and to learn C++ as I have never had made a real project before in C++.

