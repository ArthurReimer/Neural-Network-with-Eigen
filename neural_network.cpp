#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <memory>
#include <typeinfo>
#include <algorithm>
using namespace std;
#include "mnist/include/mnist/mnist_reader.hpp"
 
typedef std::vector<std::vector<float>> Matrix;

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
    for (std::size_t i = 0; i < vec.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

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

class Network {
    public:
        // Input layer is implicit, not defined explicitly
        // "Next" -> next layer right
        // "Previous" -> next layer left
        std::vector<std::unique_ptr<Layer>> layers;

        // Setting random starting weights with floats between -1 and 1
        void setWeights(const size_t inputLength) {
            std::size_t layerAmount = layers.size();
            Layer* prevLayer = nullptr;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);
            
            for (std::size_t l = 0; l < layerAmount; l++) {
                Layer& currentLayer = *(layers[l]);

                int cols = (l == 0)
                    ? inputLength
                    : layers[l-1]->neuronAmount;
                int rows = currentLayer.neuronAmount;

                currentLayer.weights = Matrix(rows, std::vector<float>(cols));

                for (std::size_t r = 0; r < rows; ++r) {
                    for (std::size_t c = 0; c < cols; ++c) {
                        currentLayer.weights[r][c] = dist(gen);
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
                Layer& currentLayer = *(layers[l]);
                std::vector<float> biases(currentLayer.neuronAmount);

                for (std::size_t n = 0; n < currentLayer.neuronAmount; n++) {
                    biases[n] = dist(gen);
                }
                currentLayer.biases = biases;
            }
        }

        // Forward pass calculation
        void forwardPass(std::vector<float> inputs) {
            std::vector<float> currentInputs = inputs;

            for (std::size_t l = 0; l < layers.size(); l++) {
                Layer& currentLayer = *(layers[l]);
                std::size_t neuronAmount = currentLayer.neuronAmount;

                std::vector<float> netInputs(neuronAmount);
                std::vector<float> activations(neuronAmount);
                
                for (std::size_t n = 0; n < currentLayer.neuronAmount; n++) {
                    float weightedSum = 0.0f;

                    for (std::size_t c = 0; c < currentLayer.weights[n].size(); c++) {
                        weightedSum += currentLayer.weights[n][c] * currentInputs[c];
                    }
                    netInputs[n] = weightedSum + currentLayer.biases[n];
                    activations[n] = sigmoid(netInputs[n]);
                }
                currentInputs = activations;
                currentLayer.netInputs = netInputs;
                currentLayer.activations = activations;

                // if (l == layers.size()-1) {
                //     for (std::size_t i = 0; i < currentLayer->activations.size(); i++) {
                //         std::cout << currentInputs[i] << "\n";
                //     }
                // }
            }
        }

        // Backwardpass calculation
        void backwardPass(const float η, std::vector<float> target, std::vector<float> inputs) {
            std::size_t layerAmount = layers.size();

            for (int l = layerAmount - 1; l >= 0; l--) {
                Layer& currentLayer = *(layers[l]);
                std::size_t neuronAmount = currentLayer.neuronAmount;
                std::vector<float> deltas(neuronAmount);

                if (l == layerAmount-1) {
                    for (std::size_t n = 0; n < neuronAmount; n++) {
                        deltas[n] = sigmoidDerivative(currentLayer.netInputs[n]) * (target[n]-currentLayer.activations[n]);
                    }
                } else {
                    Layer& nextLayer = *(layers[l+1]);

                    for (std::size_t n = 0; n < neuronAmount; n++) {
                        float weighted = 0.0f;
                        for (std::size_t next_n = 0; next_n < nextLayer.neuronAmount; next_n++) {
                            weighted += nextLayer.weights[next_n][n] * nextLayer.deltas[next_n];
                        }
                        deltas[n] = sigmoidDerivative(currentLayer.netInputs[n]) * weighted;
                    }
                }

                currentLayer.deltas = deltas;
                // printVector(currentLayer.deltas);

                for (std::size_t n = 0; n < neuronAmount; n++) {
                    std::size_t prevAmount = (l == 0)
                        ? inputs.size()
                        : layers[l-1]->neuronAmount;
                    currentLayer.biases[n] += η * deltas[n];

                    for (std::size_t prev_n = 0; prev_n < prevAmount; prev_n++) {
                        float activation = (l == 0)
                            ? inputs[prev_n]
                            : layers[l-1]->activations[prev_n];          
                        currentLayer.weights[n][prev_n] += η * activation * deltas[n];
                    }
                }
            }
        }
};



std::vector<float> normalize(const std::vector<uint8_t>& vec) {
    std::size_t len = vec.size();
    std::vector<float> newVec(len);

    for (size_t i = 0; i < len; i++) {
        newVec[i] = static_cast<float>(vec[i]) / 255.0f;
    }

    return newVec;
}

std::vector<float> oneHotEncoding(uint8_t label, size_t outputAmount) {
    vector<float> newVec(outputAmount, 0.0f);

    if (label < outputAmount) {
        newVec[label] = 1.0f;
    }

    return newVec;
}

size_t argmax(const std::vector<float>& vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

// function MSE_loss(expected, output, output_length)
//     return sum((expected-output).*(expected-output)) / output_length
// end

float lossMSE(std::vector<float> expected, std::vector<float> output) {
    if (expected.size() != output.size()) {
        std::cout << "Invalid size output and expected in MSE calculation" << "\n";
        return 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < output.size(); i++) {
        sum += (expected[i]-output[i]) * (expected[i]-output[i]);
    }
    return sum/output.size();
}



int main() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    Layer hiddenLayer; hiddenLayer.neuronAmount = 512;
    Layer hiddenLayer2; hiddenLayer2.neuronAmount = 256;
    Layer outputLayer; outputLayer.neuronAmount = 10;

    Network nn;

    const size_t inputLength = dataset.training_images[0].size();
    const size_t outputLength = outputLayer.neuronAmount;

    nn.layers.push_back(std::make_unique<Layer>(hiddenLayer));
    nn.layers.push_back(std::make_unique<Layer>(hiddenLayer2));
    nn.layers.push_back(std::make_unique<Layer>(outputLayer));

    nn.setWeights(inputLength);
    nn.setBiases();

    for (size_t e = 0; e < 1; e++) {
        size_t correctCount = 0;

        for (size_t i = 0; i < dataset.training_images.size(); i++) {
            std::vector<float> inputs = normalize(dataset.training_images[i]);
            
            uint8_t correctLabel = dataset.training_labels[i];
            std::vector<float> correctLabelArr = oneHotEncoding(correctLabel, outputLength);
            
            nn.forwardPass(inputs);
            
            int predictedLabel = argmax(nn.layers[nn.layers.size()-1]->activations);
            if (predictedLabel == correctLabel) {
                correctCount++;
            }

            std::cout << lossMSE(correctLabelArr, nn.layers[nn.layers.size()-1]->activations) << "\n";
            nn.backwardPass(0.01f, correctLabelArr, inputs);
        }

        float accuracy = static_cast<float>(correctCount) / dataset.training_images.size();
        std::cout << "Epoch " << e + 1 << " Accuracy: " << accuracy * 100.0f << "%\n";
    }


    std::cout << "Ended" << "\n";



    // Layer hiddenLayer; hiddenLayer.neuronAmount = 256;
    // Layer hiddenLayer2; hiddenLayer2.neuronAmount = 256;
    // Layer outputLayer; outputLayer.neuronAmount = 5;

    // std::vector<float> inputs = {0.5, 0.1, 0.2, 0.3, 0.6, 0.7, 0.9, 0.8};

    // Network nn;

    // nn.layers.push_back(std::make_unique<Layer>(hiddenLayer));
    // nn.layers.push_back(std::make_unique<Layer>(outputLayer));

    // nn.setWeights(inputs);
    // nn.setBiases();

    // auto start = std::chrono::high_resolution_clock::now();
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> dist(0.0, 9.0);

    // for (int i = 1; i < 100; i++) {
    //     std::vector<float> inputs = {dist(gen), dist(gen), dist(gen), dist(gen), dist(gen)};
    //     nn.forwardPass(inputs);
    //     nn.backwardPass(0.01f, {0,0,0,0,1}, inputs);
    // }

    // auto end = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    // std::cout << "Time taken: " << duration.count() << " seconds" << "\n";
    
    return 0;
}