#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
using namespace std;

typedef std::vector<std::vector<float>> Matrix;

class Layer {
    public:
    int neuronAmount;
    Matrix weights;
    std::vector<float> biases;
    std::vector<float> activations;
    std::vector<float> netInputs;
        
    Layer() : neuronAmount(0) {}
};

class Network {
    public:
        std::vector<Layer> layers;

        // Setting random starting weights with floats between -1 and 1
        void setWeights(std::vector<float> inputs) {
            size_t layerAmount = layers.size();
            Layer* prevLayer = nullptr;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);
            
            for (size_t l = 0; l < layerAmount; l++) {
                Layer* currentLayer = &layers[l];

                int cols = (l == 0)
                    ? inputs.size()
                    : layers[l-1].neuronAmount;
                int rows = currentLayer->neuronAmount;

                currentLayer->weights = Matrix(rows, std::vector<float>(cols));

                for (size_t r = 0; r < rows; ++r) {
                    for (size_t c = 0; c < cols; ++c) {
                        currentLayer->weights[r][c] = dist(gen);
                    } 
                }
            }
        }

        // Setting random starting biases with floats between -1 and 1
        void setBiases() {
            size_t layerAmount = layers.size();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);

            for (size_t l = 0; l < layerAmount; l++) {
                Layer* currentLayer = &layers[l];
                currentLayer->biases.resize(currentLayer->neuronAmount);

                for (size_t n = 0; n < currentLayer->neuronAmount; n++) {
                    currentLayer->biases[n] = dist(gen);
                }
            }
        }

        // Forward pass calculation of all layers in the neural network
        void forwardPass(std::vector<float> inputs) {
            std::vector<float> currentInputs = inputs;

            for (size_t l = 0; l < layers.size(); l++) {
                Layer* currentLayer = &layers[l];
                currentLayer->activations.resize(currentLayer->neuronAmount);
                currentLayer->netInputs.resize(currentLayer->neuronAmount);
                
                for (size_t n = 0; n < currentLayer->neuronAmount; n++) {
                    float weightedSum = 0.0f;

                    for (size_t c = 0; c < currentLayer->weights[n].size(); c++) {
                        weightedSum += currentLayer->weights[n][c] * currentInputs[c];
                    }
                    currentLayer->netInputs[n] = weightedSum + currentLayer->biases[n];
                    currentLayer->activations[n] = std::max(0.0f, currentLayer->netInputs[n]);
                }
                currentInputs = currentLayer->activations;
            }
        }
};

void testForwardPass() {
    Layer hiddenLayer, outputLayer;
    hiddenLayer.neuronAmount = 2;  // 2 hidden neurons
    outputLayer.neuronAmount = 1;  // 1 output neuron

    Network nn;
    nn.layers = {hiddenLayer, outputLayer};  // Input layer is implicit, not defined explicitly

    nn.layers[0].weights = {{0.5, -0.5}, {0.1, -0.3}};  // Weights for hidden layer (2x2 matrix)
    nn.layers[0].biases = {0.1, -0.2};  // Biases for hidden layer
    nn.layers[1].weights = {{0.4, -0.1}};  // Weights for output layer (1x2 matrix)
    nn.layers[1].biases = {0.3};  // Biases for output layer

    std::vector<float> inputs = {0.5, -0.5};  // 2 inputs

    nn.forwardPass(inputs);

    cout << nn.layers[1].activations[0] << "\n"; // Correct output should be 0.54
}


int main() {
    Layer hiddenLayer; hiddenLayer.neuronAmount = 64;
    Layer outputLayer; outputLayer.neuronAmount = 10;

    std::vector<Layer> layers = {hiddenLayer, outputLayer};
    std::vector<float> inputs = {0.5, 0.1, 0.2, 0.3, 0.6, 0.7, 0.9, 0.8};

    Network nn;
    nn.layers = layers;
    nn.setWeights(inputs);
    nn.setBiases();
    nn.forwardPass(inputs);

    return 0;
}