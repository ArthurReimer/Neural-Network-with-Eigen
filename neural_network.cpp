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

class Network {
    public:
        // Input layer is implicit, not defined explicitly
        // "Next" -> next layer right
        // "Previous" -> next layer left
        std::vector<std::unique_ptr<Layer>> layers;
        const int batchSize;
        int inputLength;

        Network(int b) : 
            batchSize(b) {}
        
        // Adds an layer to layers
        void addLayer(const int n) {
            this->layers.push_back(std::make_unique<Layer>(n, batchSize));
        }

        // Setup of biases and weights. Should only be executed once
        void setup(const int inputLen) {
            inputLength = inputLen;
            this->setWeights();
            this->setBiases();
        }

        // Setting random starting weights with floats between -1 and 1
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

        // Setting random starting biases with floats between -1 and 1
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

        // // Forward pass calculation
        void forwardPass(const MatrixXf& inputs) {
            MatrixXf currentInputs = inputs;

            for (int l = 0; l < layers.size(); l++) {
                Layer& currentLayer = *(layers[l]);
                int neuronAmount = currentLayer.neuronAmount;

                currentLayer.netInputs = currentLayer.weights * currentInputs;
                currentLayer.netInputs.colwise() += currentLayer.biases;
                currentLayer.activations = act::sigmoid(currentLayer.netInputs);

                currentInputs = currentLayer.activations;
            }
        }

        // Backwardpass calculation
        void backwardPass(const float η, const MatrixXf& target, const MatrixXf& inputs) {
            int layerAmount = layers.size();
            Eigen::setNbThreads(1);

            for (int l = layerAmount - 1; l >= 0; l--) {
                Layer& currentLayer = *(layers[l]);
                int neuronAmount = currentLayer.neuronAmount;
                currentLayer.sigmoidDerivs = act::derivSigmoid(currentLayer.activations);
                
                if (l == layerAmount-1) {
                    currentLayer.deltas = currentLayer.sigmoidDerivs.array() * (target.array() - currentLayer.activations.array());
                } else {
                    Layer& nextLayer = *(layers[l+1]);
                    
                    currentLayer.weightedDeltas = nextLayer.weights.transpose() * nextLayer.deltas;
                    currentLayer.deltas = currentLayer.sigmoidDerivs.array() * currentLayer.weightedDeltas.array();
                }

                MatrixXf prevActivations = (l == 0) ? inputs : layers[l-1]->activations;
                currentLayer.biases += η * currentLayer.deltas.rowwise().mean();
                currentLayer.weights.noalias() += η * currentLayer.deltas * prevActivations.transpose().eval() / batchSize;
            }
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
    const int epoches = 30;

    // Mutable variables
    MatrixXf inputs;
    inputs.resize(784, batchSize);
    std::vector<std::vector<uint8_t>> rawInputs;
    std::vector<uint8_t> labels(batchSize);

    // Network Setup
    Network nn(batchSize);
    nn.addLayer(48);
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
}

int main() {
    Eigen::setNbThreads(4);
    std::cout << "Program started\n";
    train();
    std::cout << "Training finished\n";
    return 0;
}



#endif