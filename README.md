# C++ Neural Network with Eigen from scratch
##  Libraries used
- MNIST reader (MNIST data included)
- Eigen
- Standard Library (C++)
## Description
I made this simple neural network in C++ by utilizing ```Eigen's``` fast matrix & vector calculation.
I chose to create a neural network from scratch in C++ so that I can better understand how these work and can learn actual performance & accuracy strategies.
By making this project I gained lots of experience in both C++ (in which I've never had a real project before) and neural networks.
Currently only a handful of activation functions are supported, but I'm planning on adding more in the future.
Because Eigen uses the CPU mainly, GPU is not yet supported, even though GPU would offer great performance benefits with multithreading and multiprocessing.

## Usage:
### Creating the network
The input layer is implicit and is not represented by a layer.

``` Cpp
// import lib
#include "neural_network.hpp"

// ...
void train() {
    // MNIST dataset
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../external/mnist");

    

    // Constants
    const int inputLength = dataset.training_images[0].size();
    const int outputLength = 10;
    const int epochs = 100;

    double learningRate = 0.0005;

    // Network Setup
    NeuralNetwork nn(inputLength);
    nn.addLayer(128, LEAKY_RELU);
    nn.addLayer(128, LEAKY_RELU);
    nn.addLayer(outputLength, SIGMOID);

    VectorXf inputs(inputLength);
    
    Timer timer = Timer();
    timer.startTimer();

    std::cout << "Neural Network Model" << "\n"
            << "  Denselayer amount - " << nn.layers.size() << "\n"
            << "  Weight amount - " << nn.getWeightAmount() << "\n"
            << "  Bias amount - " << nn.getBiasAmount() << "\n\n";

    std::cout << "Starting Training" << "\n";


    for (int e = 0; e < epochs; e++) {
        int correct = 0;
        for (int i = 0; i < dataset.training_images.size(); i++) {
            
            normalizeInto(inputs, dataset.training_images[i]);
            nn.forwardpass(inputs);

            int label = static_cast<int>(dataset.training_labels[i]);
            int prediction = argmax(nn.layers[nn.layers.size()-1].activations);
            if (prediction == label) {
                correct += 1;
            }


            VectorXf target = oneHotEncoding(label, outputLength);
            nn.backpropagation(inputs, learningRate, target);
        }
        float accuracy = static_cast<float>(correct) / dataset.training_images.size();
        std::cout << "Epoch " << e+1 <<" Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "% ";
        timer.endTimerSeconds();
    }
}
}
```
### Forwardpass + backwardpass
To utilize the forward- and backwardpass you just have to call their function.
An actual training example is located in the ```mnist_training.cpp``` script and above with MNIST.

## Compiling
To achieve good performance, you need to use optimization flags.
This would be an example of how to compile the training file: <br />
``` g++ -O3 -fopenmp mnist_training.cpp -I../src -I../external/eigen -I../external/mnist/include -std=c++17 -o mnist_training.exe 2>&1 ```


## Example output data on MNIST

### Output
``` console
Neural Network Model
  Denselayer amount - 3
  Weight amount - 118016
  Bias amount - 266

Starting Training
Epoch 1 Accuracy: 70.61% Elapsed time: 1sec
Epoch 2 Accuracy: 88.25% Elapsed time: 3sec
Epoch 3 Accuracy: 90.24% Elapsed time: 4sec
Epoch 4 Accuracy: 91.22% Elapsed time: 6sec
Epoch 5 Accuracy: 91.94% Elapsed time: 7sec
Epoch 6 Accuracy: 92.49% Elapsed time: 9sec
Epoch 7 Accuracy: 92.92% Elapsed time: 11sec
Epoch 8 Accuracy: 93.32% Elapsed time: 12sec
Epoch 9 Accuracy: 93.65% Elapsed time: 13sec
Epoch 10 Accuracy: 93.94% Elapsed time: 15sec
Epoch 11 Accuracy: 94.20% Elapsed time: 16sec
Epoch 12 Accuracy: 94.43% Elapsed time: 18sec
Epoch 13 Accuracy: 94.65% Elapsed time: 19sec
Epoch 14 Accuracy: 94.86% Elapsed time: 21sec
Epoch 15 Accuracy: 95.03% Elapsed time: 22sec
Epoch 16 Accuracy: 95.19% Elapsed time: 25sec
Epoch 17 Accuracy: 95.33% Elapsed time: 27sec
Epoch 18 Accuracy: 95.46% Elapsed time: 29sec
Epoch 19 Accuracy: 95.60% Elapsed time: 32sec
Epoch 20 Accuracy: 95.72% Elapsed time: 35sec
[...]
Epoch 100 Accuracy: 98.83% Elapsed time: 194sec
```
