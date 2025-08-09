# C++ Neural Network with Eigen from scratch
##  Libraries:
- MNIST
- Eigen
- Standard Library (C++)
## Description:
I made this simple neural network in C++ by utilizing Eigen's fast matrix & vector calculation.
I chose to create a neural network from scratch in C++ so that I can better understand how these work and can learn actual performance & accuracy strategies.
Currently only a handful of activation functions are supported, but im planning on adding more for the future.
Because Eigen uses CPU mainly, GPU is not yet supported even though GPU would offer great performance benefits with multithreading and multiprocessing.
## Usage:
### Creating the network
``` Cpp
// ...

#include "neural_network.hpp"

int main() {
  // Defines how many input vectors get processed at the same time
  const int batch_size = 10;

  // Creating the nn
  NN::Network nn(batch_size);

  // Adding layers with neurons & activation function
  nn.addLayer(32, LEAKY_RELU);
  nn.addLayer(32, LEAKY_RELU);
  nn.addLayer(outputLength, LEAKY_RELU);

  // Mandatory setup function that has to be called after adding all layers
  nn.setup(inputLength);

  // Training... (see 'mnist_training' script for training example)

  return 0;
}
```
### Forwardpass + backwardpass
To utilize the forward- and backwardpass you just have to call their function.
Actual training example is located in the 'mnist_training' script. The following code is simplified
``` Cpp 
// ...

#include "neural_network.hpp"

int main() {
  // Defines how many input vectors get processed at the same time
  const int batch_size = 10;

  // Creating the nn
  NN::Network nn(batch_size);

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

