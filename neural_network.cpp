#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <memory>
#include <typeinfo>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include "mnist/include/mnist/mnist_reader.hpp"
using namespace std;

#include "external/eigen/Eigen/Dense"
using Eigen::MatrixXf;
using Eigen::VectorXf;

#include "activations.h"
using namespace act;

#ifndef EIGEN_CORE_H

#else

inline void printVector(const VectorXf& vec) {
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

inline void printVector_uint8_t(const std::vector<uint8_t>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << static_cast<int>(vec[i]);
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}


/*
The layer class contains all of the important mutable variables like the weights and biases.
Weights and biases are initialized after all the layers are pushed into the network and .setup() is called once.
*/
class Layer {
    public:
        int neuronAmount;
        MatrixXf weights;
        VectorXf biases;
        MatrixXf activations;
        MatrixXf netInputs;
        MatrixXf deltas;
        MatrixXf sigmoidDerivs;
        MatrixXf weightedDeltas;

        Layer(int n, int batchSize) : neuronAmount(n),
            biases(VectorXf::Zero(n)),
            activations(MatrixXf::Zero(n, batchSize)),
            netInputs(MatrixXf::Zero(n, batchSize)),
            deltas(MatrixXf::Zero(n, batchSize)),
            sigmoidDerivs(MatrixXf::Zero(n, batchSize)),
            weightedDeltas(MatrixXf::Zero(n, batchSize)) {}
};

/*
The network class is the neural network. It contains all of the layers and most of the logic.
.addLayer pushes a new layer into the Network. This is the meant way to add another layer to the network.
After all layers were pushed into the neural network, .setup() should be called once to create all the weights and biases.
Not doing so results in undefined behaviours.
Currently no multithreaded batching is implemented, only batching, and no GPU support is implemented yet.
*/
class Network {
    public:
        /*
        Input layer is implicit, not defined explicitly
        Layers is a vector of smart pointers(unique_ptrs)
        */

        std::vector<std::unique_ptr<Layer>> layers;
        const int batchSize;
        int inputLength;

        /*
        Constructor for the network object
        */
        Network(int b) : 
            batchSize(b) {}
        
        /*
        Pushes an layer to layers
        Input layer is implicit
        */
        void addLayer(const int n) {
            this->layers.push_back(std::make_unique<Layer>(n, batchSize));
        }


        /*
        Setup of biases and weights
        Should only be executed once
        */
        void setup(const int inputLen) {
            inputLength = inputLen;
            this->setWeights();
            this->setBiases();
        }

        /*
        Setting random starting weights
        Weights is a matrix of type float
        */
        void setWeights() {
            int layerAmount = layers.size();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-0.5, 0.5);
            
            for (int l = 0; l < layerAmount; l++) {
                Layer& currentLayer = *(layers[l]);

                int cols = (l == 0)
                    ? inputLength
                    : layers[l-1]->neuronAmount;
                int rows = currentLayer.neuronAmount;
        
                currentLayer.weights = MatrixXf(rows, cols);

                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        currentLayer.weights(r, c) = dist(gen);
                    } 
                }
            }
        }

        /*
        Setting random starting biases.
        Biases is a vector of type float.
        */
        void setBiases() {
            int layerAmount = layers.size();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-0.5, 0.5);

            for (int l = 0; l < layerAmount; l++) {
                Layer& currentLayer = *(layers[l]);
                std::vector<float> biases(currentLayer.neuronAmount);

                for (int n = 0; n < currentLayer.neuronAmount; ++n) {
                    currentLayer.biases(n) = dist(gen);
                }
            }
        }

        /*
        Forwardpass calculation through all layers of the network.
        Inputs is a type of matrix so that batching is possible.

        */
        void forwardPass(const MatrixXf& inputs) {
            /*
            Current inputs is changed after the layer calcuation so that the next layer already has their inputs
            */
            MatrixXf currentInputs = inputs;

            /*
            Looping through all layers. All layers are depedent on eachother for the calculation
            */
            for (int l = 0; l < layers.size(); l++) {
                Layer& currentLayer = *(layers[l]);
                int neuronAmount = currentLayer.neuronAmount;

                /*
                Calcuation of the net inputs for the current layer
                Inorder for the dimensions to align with the weighted inputs and the biases, column wise addition is required (colwise())
                */
                currentLayer.netInputs = currentLayer.weights * currentInputs;
                currentLayer.netInputs.colwise() += currentLayer.biases;

                /*
                Activations are calculated using sigmoid. In the future there should be more options
                */
                currentLayer.activations = act::sigmoid(currentLayer.netInputs);

                /*
                Updating the current inputs for the next layer
                */
                currentInputs = currentLayer.activations;
            }
        }

        /*
        Backwardpass calculation through all layers of the network at reverse.
        Inputs is a type of matrix so that batching can be used and multiple inputs can be calculated in one chunk.
        η is the learning rate of the backwardpass.
        
        Eigens threading has to be disabled by setting a max of one thread duo to threading problems when updating weights/biases.
        After the backwardpass is done the max thread size will be updated back to the original thread size.
        */
        void backwardPass(const float η, const MatrixXf& target, const MatrixXf& inputs) {
            /*
            Disabling mutli threading by allowing a maxiumum of one thread.
            */
            Eigen::setNbThreads(1);
            int layerAmount = layers.size();

            for (int l = layerAmount - 1; l >= 0; l--) {
                Layer& currentLayer = *(layers[l]);
                int neuronAmount = currentLayer.neuronAmount;

                /*
                Calculating all the derivative activations for the current layer
                */
                currentLayer.sigmoidDerivs = act::derivSigmoid(currentLayer.activations);
                
                if (l == layerAmount-1) {
                    /*
                    Output layer delta (gradient) calculation
                    .array() has to be used for element wise calculation
                    */
                    currentLayer.deltas = currentLayer.sigmoidDerivs.array() * (target.array() - currentLayer.activations.array());
                } else {
                    /*
                    Hidden layer delta (gradient) calculation
                    Dereferencing the next layer smart pointer
                    .array() has to be used for element wise calculation
                    "nextLayer" is the layer right to the current layer
                    */
                    Layer& nextLayer = *(layers[l+1]);
                    
                    currentLayer.weightedDeltas = nextLayer.weights.transpose() * nextLayer.deltas;
                    currentLayer.deltas = currentLayer.sigmoidDerivs.array() * currentLayer.weightedDeltas.array();
                }

                /*
                Updating all biases & weights for the current layer
                .noalias speeds up the weight updating significantly
                Because the backwardpass uses batching, the weights gradient has to be divided by the batch size
                "previous" is refering to the layer left of the current layer.
                Duo to inputs being implicit, the previous layer at the index 0 would require the inputs argument instead of the previous layer
                */
                MatrixXf prevActivations = (l == 0) ? inputs : layers[l-1]->activations;
                currentLayer.biases += η * currentLayer.deltas.rowwise().mean();
                currentLayer.weights.noalias() += η * currentLayer.deltas * prevActivations.transpose().eval() / batchSize;
            }

            /*
            Setting back the max thread count
            */
            Eigen::setNbThreads(12);
        }
};

inline void normalizeInto(Eigen::MatrixXf& mat, const std::vector<std::vector<uint8_t>>& input) {
    const int rows = mat.rows();
    const int cols = mat.cols();

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            mat(row, col) = static_cast<float>(input[col][row]) / 255.0f;
        }
    }
}

inline MatrixXf oneHotEncoding(vector<uint8_t> label, int outputAmount, const int batchSize) {
    MatrixXf newMat = Eigen::MatrixXf::Zero(outputAmount, batchSize);

    for (int i = 0; i < batchSize; i++) {
        newMat(label[i],i) = 1.0f;
    }
    return newMat;
}

inline int argmax(const VectorXf& vec) {
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return static_cast<int>(maxIndex);
}

inline float lossMSE(const MatrixXf& expected, const MatrixXf& output) {
    return ((expected - output).array().square().sum()) / expected.cols();
}

void train() {
    // MNIST dataset
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    // Constants
    const int inputLength = dataset.training_images[0].size();
    const int outputLength = 10;
    const float learningRate = 0.06f;
    const int batchSize = 8;
    const int epoches = 5;

    // Mutable variables
    MatrixXf inputs;
    inputs.resize(784, batchSize);
    std::vector<std::vector<uint8_t>> rawInputs;
    std::vector<uint8_t> labels(batchSize);

    // Network Setup
    Network nn(batchSize);
    nn.addLayer(258);
    nn.addLayer(258);
    nn.addLayer(outputLength);
    nn.setup(inputLength);

    for (int e = 0; e < epoches; e++) {
        int numBatches = dataset.training_images.size() / batchSize;
        float epochLoss = 0.0f;
        int correctPredictions = 0;
        auto start = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < numBatches; b++) {
            rawInputs.clear();

            for (int i = 0; i < batchSize; i++) {
                int index = b * batchSize + i;
                rawInputs.push_back(dataset.training_images[index]);
                labels[i] = dataset.training_labels[index];
            }

            normalizeInto(inputs, rawInputs);
            MatrixXf targets = oneHotEncoding(labels, outputLength, batchSize);

            // Forwardpass
            nn.forwardPass(inputs);

            // Get output from last layer
            const MatrixXf& predictions = nn.layers.back()->activations;

            // Compute batch loss
            epochLoss += lossMSE(targets, predictions);

            // Compute accuracy
            for (int i = 0; i < batchSize; i++) {
                VectorXf col = predictions.col(i);
                int predicted = argmax(col);
                if (predicted == labels[i]) {
                    correctPredictions++;
                }
            }

            // Backpropagation
            nn.backwardPass(learningRate, targets, inputs);
        }

        float avgLoss = epochLoss / numBatches;
        float accuracy = static_cast<float>(correctPredictions) / (numBatches * batchSize);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Epoch " << (e + 1) << "/" << epoches
                << " | Loss: " << avgLoss
                << " | Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%"
                << " | Time taken: " << duration.count() << " seconds" << "" << std::endl;
    }

    // Test the trained nn
    int correctTestPredictions = 0;
    const int testSize = dataset.test_images.size();
    MatrixXf testInput(784, 1);

    for (int i = 0; i < testSize; ++i) {
        for (int j = 0; j < 784; ++j) {
            testInput(j, 0) = static_cast<float>(dataset.test_images[i][j]) / 255.0f;
        }

        nn.forwardPass(testInput);
        const MatrixXf& output = nn.layers.back()->activations;
        int predicted = argmax(output.col(0));
        int actual = dataset.test_labels[i];

        if (predicted == actual) {
            ++correctTestPredictions;
        }
    }

    float testAccuracy = static_cast<float>(correctTestPredictions) / testSize;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2)
              << testAccuracy * 100.0f << "%" << std::endl;

}

int main() {
    Eigen::setNbThreads(12);
    std::cout << "Program started\n";
    train();
    std::cout << "Training finished\n";
    return 0;
}



#endif