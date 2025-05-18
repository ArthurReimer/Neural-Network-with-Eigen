#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <iomanip>
using namespace std;

typedef std::vector<std::vector<float>> Matrix;

class Layer {
    public:
    int neuronAmount;
    Matrix weights;
    std::vector<float> biases;
    std::vector<float> activations;
    std::vector<float> netInputs;
    std::vector<float> deltas;
        
    Layer() : neuronAmount(0) {}
};

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Derivative of the sigmoid function
float sigmoidDerivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

void printVector(const std::vector<float>& vec) {
    std::cout << "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

std::vector<float> flatten(Matrix mat) {
    std::size_t rows = mat.size();
    std::size_t columns = mat.empty() ? 0 : mat[0].size();
    std::vector<float> newVec(rows*columns);

    for (std::size_t r = 0; r < rows; r++) {
        for (std::size_t c = 0; c < columns; c++) {
            newVec[r * columns + c] = mat[r][c];
        }
    }
    return newVec;
}

class Network {
    public:
        // Input layer is implicit, not defined explicitly
        // "Next" -> next layer right
        // "Previous" -> next layer left
        std::vector<Layer> layers;

        // Setting random starting weights with floats between -1 and 1
        void setWeights(std::vector<float> inputs) {
            std::size_t layerAmount = layers.size();
            Layer* prevLayer = nullptr;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);
            
            for (std::size_t l = 0; l < layerAmount; l++) {
                Layer* currentLayer = &layers[l];

                int cols = (l == 0)
                    ? inputs.size()
                    : layers[l-1].neuronAmount;
                int rows = currentLayer->neuronAmount;

                currentLayer->weights = Matrix(rows, std::vector<float>(cols));

                for (std::size_t r = 0; r < rows; ++r) {
                    for (std::size_t c = 0; c < cols; ++c) {
                        currentLayer->weights[r][c] = dist(gen);
                    } 
                }
            }
        }

        // Setting random starting biases with floats between -1 and 1
        void setBiases() {
            std::size_t layerAmount = layers.size();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);

            for (std::size_t l = 0; l < layerAmount; l++) {
                Layer* currentLayer = &layers[l];
                currentLayer->biases.resize(currentLayer->neuronAmount);

                for (std::size_t n = 0; n < currentLayer->neuronAmount; n++) {
                    currentLayer->biases[n] = dist(gen);
                }
            }
        }

        // Forward pass calculation of all layers in the neural network
        void forwardPass(std::vector<float> inputs) {
            std::vector<float> currentInputs = inputs;

            for (std::size_t l = 0; l < layers.size(); l++) {
                Layer* currentLayer = &layers[l];
                std::size_t neuronAmount = currentLayer->neuronAmount;

                std::vector<float> netInputs(neuronAmount);
                std::vector<float> activations(neuronAmount);
                
                for (std::size_t n = 0; n < currentLayer->neuronAmount; n++) {
                    float weightedSum = 0.0f;

                    for (std::size_t c = 0; c < currentLayer->weights[n].size(); c++) {
                        weightedSum += currentLayer->weights[n][c] * currentInputs[c];
                    }
                    netInputs[n] = weightedSum + currentLayer->biases[n];
                    activations[n] = sigmoid(netInputs[n]);
                }
                currentInputs = activations;
                currentLayer->netInputs = netInputs;
                currentLayer->activations = activations;

                // if (l == layers.size()-1) {
                //     for (std::size_t i = 0; i < currentLayer->activations.size(); i++) {
                //         std::cout << currentInputs[i] << "\n";
                //     }
                // }
            }
        }

        // Backward pass calculation
        void backwardPass(const float η, std::vector<float> target, std::vector<float> inputs) {
            std::size_t layerAmount = layers.size();

            for (int l = layerAmount - 1; l >= 0; l--) {
                Layer *currentLayer = &layers[l];
                std::size_t neuronAmount = currentLayer->neuronAmount;
                std::vector<float> deltas(neuronAmount);

                if (l == layerAmount-1) {
                    for (std::size_t n = 0; n < neuronAmount; n++) {
                        deltas[n] = sigmoidDerivative(currentLayer->netInputs[n]) * (target[n]-currentLayer->activations[n]);
                    }
                } else {
                    Layer *nextLayer = &layers[l+1];

                    for (std::size_t n = 0; n < neuronAmount; n++) {
                        float weighted = 0.0f;
                        for (std::size_t next_n = 0; next_n < nextLayer->neuronAmount; next_n++) {
                            weighted += nextLayer->weights[next_n][n] * nextLayer->deltas[next_n];
                        }
                        deltas[n] = sigmoidDerivative(currentLayer->netInputs[n]) * weighted;
                    }
                }

                printVector(deltas);
                currentLayer->deltas = deltas;

                for (std::size_t n = 0; n < neuronAmount; n++) {
                    std::size_t prevAmount = (l == 0)
                        ? inputs.size()
                        : layers[l-1].neuronAmount;
                    currentLayer->biases[n] -= η * deltas[n];

                    for (std::size_t prev_n = 0; prev_n < prevAmount; prev_n++) {
                        float activation = (l == 0)
                            ? inputs[prev_n]
                            : layers[l-1].activations[prev_n];          
                        currentLayer->weights[n][prev_n] -= η * activation * deltas[n];
                    }
                }
            }
        }
};

void testForwardPass() {
    Layer hiddenLayer, outputLayer;
    hiddenLayer.neuronAmount = 2;  // 2 hidden neurons
    outputLayer.neuronAmount = 1;  // 1 output neuron



    Network nn;
    nn.layers = {hiddenLayer, outputLayer};  // Input layer is implicit, not defined explicitly

    nn.layers[0].weights = {{0.5, -0.5}, {0.1, -0.3}};  // Weights for hidden layer (2x2 Matrix)
    nn.layers[0].biases = {0.1, -0.2};  // Biases for hidden layer
    nn.layers[1].weights = {{0.4, -0.1}};  // Weights for output layer (1x2 Matrix)
    nn.layers[1].biases = {0.3};  // Biases for output layer

    std::vector<float> inputs = {0.5, -0.5};  // 2 inputs

    std::cout << "Starting" << "\n";
    nn.forwardPass(inputs);

    cout << nn.layers[1].activations[0] << "\n"; // Correct output should be 0.624
}


int main() {
    Layer hiddenLayer; hiddenLayer.neuronAmount = 128;
    Layer hiddenLayer2; hiddenLayer2.neuronAmount = 128;
    Layer outputLayer; outputLayer.neuronAmount = 5;

    std::vector<float> inputs = {0.5, 0.1, 0.2, 0.3, 0.6, 0.7, 0.9, 0.8};

    Network nn;
    nn.layers = {hiddenLayer, hiddenLayer2, outputLayer};;
    nn.setWeights(inputs);
    nn.setBiases();
    nn.forwardPass(inputs);
    nn.backwardPass(0.5f, {5.0, 3.0, 1.0, 3.0, 1.0}, inputs);

    // for (int i = 0; i < 1000; i++) {
        
    // }
    
    return 0;
}