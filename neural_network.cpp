#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
using namespace std;

class Layer {
    public:
    int neuronAmount;
    float* weights;
    float* biases;
        
    Layer() : neuronAmount(0), weights(nullptr), biases(nullptr) {}

    ~Layer() {
        if (weights != nullptr) {
            free(weights);
        }
        if (biases != nullptr) {
            free(biases);
        }
    }
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
            
            for (int i = 0; i < layerAmount; i++) {
                Layer* currentLayer = &layers[i];

                if (i == 0) {
                    currentLayer->weights = (float*) malloc(currentLayer->neuronAmount*inputs.size() * sizeof(float));
                    for (size_t n = 0; n < currentLayer->neuronAmount*inputs.size(); n++) {
                        currentLayer->weights[n] = dist(gen);
                    }
                } else {
                    currentLayer->weights = (float*) malloc(currentLayer->neuronAmount * prevLayer->neuronAmount * sizeof(float));
                    for (size_t n = 0; n < currentLayer->neuronAmount*prevLayer->neuronAmount; n++) {
                        currentLayer->weights[n] = dist(gen);         
                    }
                }

                prevLayer = currentLayer;
            }
        }

        // Setting random starting biases with floats between -1 and 1
        void setBiases() {
            size_t layerAmount = layers.size();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);

            for (size_t i = 0; i < layerAmount; i++) {
                Layer* currentLayer = &layers[i];
                currentLayer->biases = (float*) malloc(currentLayer->neuronAmount * sizeof(float));

                for (size_t n = 0; n < currentLayer->neuronAmount; n++) {
                    currentLayer->biases[n] = dist(gen);
                }
            }
        }

        // Forward pass calculation of all layers in the neural network
        void forwardPass(std::vector<float> inputs) {

        }
};


int main() {
    Layer hiddenLayer;
    Layer outputLayer;

    hiddenLayer.neuronAmount = 16;
    outputLayer.neuronAmount = 10;

    std::vector<Layer> layers = {hiddenLayer, outputLayer};
    std::vector<float> inputs = {0.5, 0.1, 0.2, 0.3, 0.6, 0.7, 0.9, 0.8};

    Network nn;
    nn.layers = layers;
    nn.setWeights(inputs);
    nn.setBiases();

    return 0;
}