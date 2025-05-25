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
using namespace std;
#include "mnist/include/mnist/mnist_reader.hpp"
#include "external/eigen/Eigen/Dense"
using Eigen::MatrixXf;
using Eigen::VectorXf;
 
typedef std::vector<std::vector<float>> mat;

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Derivative of the sigmoid function
float sigmoidDerivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

void printVector(const VectorXf& vec) {
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
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
    MatrixXf weights;
    VectorXf biases;
    VectorXf activations;
    VectorXf netInputs;
    VectorXf deltas;

    Layer(int n) : neuronAmount(n),
               biases(VectorXf::Zero(n)),
               activations(VectorXf::Zero(n)),
               netInputs(VectorXf::Zero(n)),
               deltas(VectorXf::Zero(n)) {}
};

class Network {
    public:
        // Input layer is implicit, not defined explicitly
        // "Next" -> next layer right
        // "Previous" -> next layer left
        std::vector<std::unique_ptr<Layer>> layers;

        // Setting random starting weights with floats between -1 and 1
        void setWeights(const size_t inputLength) {
            int layerAmount = layers.size();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1.0, 1.0);
            
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
            std::uniform_real_distribution<> dist(-1.0, 1.0);

            for (int l = 0; l < layerAmount; l++) {
                Layer& currentLayer = *(layers[l]);
                std::vector<float> biases(currentLayer.neuronAmount);

                for (int n = 0; n < currentLayer.neuronAmount; ++n) {
                    currentLayer.biases(n) = dist(gen);
                }
            }
        }

        // Forward pass calculation
        void forwardPass(const VectorXf& inputs) {
            VectorXf currentInputs = inputs;

            for (int l = 0; l < layers.size(); l++) {
                Layer& currentLayer = *(layers[l]);
                int neuronAmount = currentLayer.neuronAmount;

                currentLayer.netInputs = currentLayer.weights * currentInputs + currentLayer.biases;
                currentLayer.activations = currentLayer.netInputs.unaryExpr([](float x) { return sigmoid(x); });

                currentInputs = currentLayer.activations;
                // printVector(currentLayer.activations);
            }
        }

        // Backwardpass calculation
        void backwardPass(const float η, const VectorXf& target, const VectorXf& inputs) {
            int layerAmount = layers.size();

            int maxNeurons = 0;
            for (const auto& layer : layers) {
                maxNeurons = std::max(maxNeurons, layer->neuronAmount);
            }
            VectorXf weightedDeltas(maxNeurons);
            VectorXf sigmoidDerivs(maxNeurons);
            
            for (int l = layerAmount - 1; l >= 0; l--) {
                Layer& currentLayer = *(layers[l]);
                int neuronAmount = currentLayer.neuronAmount;

                if (sigmoidDerivs.size() != neuronAmount && weightedDeltas.size() != neuronAmount) {
                    sigmoidDerivs.resize(neuronAmount);
                    weightedDeltas.resize(neuronAmount);
                }
                    
                if (l == layerAmount-1) {
                    sigmoidDerivs = currentLayer.netInputs.unaryExpr([](float x) {
                        float s = 1.0f / (1.0f + std::exp(-x));
                        return s * (1.0f - s);
                    });
                    currentLayer.deltas = sigmoidDerivs.array() * (target.array() - currentLayer.activations.array());
                } else {
                    Layer& nextLayer = *(layers[l+1]);
                    sigmoidDerivs = currentLayer.netInputs.unaryExpr([](float x) {
                        float s = 1.0f / (1.0f + std::exp(-x));
                        return s * (1.0f - s);
                    });

                    weightedDeltas = nextLayer.weights.transpose() * nextLayer.deltas;
                    currentLayer.deltas = sigmoidDerivs.array() * weightedDeltas.array();
                }

                VectorXf prevActivations = (l == 0) ? inputs : layers[l-1]->activations;
                currentLayer.biases += η * currentLayer.deltas;
                currentLayer.weights.noalias() += η * currentLayer.deltas * prevActivations.transpose();
            }
        }
};

void normalizeInto(VectorXf& vec, const std::vector<uint8_t>& input) {
    for (int i = 0; i < input.size(); ++i) {
        vec[i] = static_cast<float>(input[i]) / 255.0f;
    }
}

VectorXf oneHotEncoding(uint8_t label, int outputAmount) {
    VectorXf newVec = Eigen::VectorXf::Zero(outputAmount);

    if (label < outputAmount) {
        newVec[label] = 1.0f;
    }

    return newVec;
}

int argmax(const VectorXf& vec) {
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return static_cast<int>(maxIndex);
}

// function MSE_loss(expected, output, output_length)
//     return sum((expected-output).*(expected-output)) / output_length
// end

float lossMSE(const VectorXf& expected, const VectorXf& output) {
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

void testPerformance() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    Layer hiddenLayer(256);
    Layer hiddenLayer2(256);
    Layer outputLayer(10);

    Network nn;

    const size_t inputLength = dataset.training_images[0].size();
    const size_t outputLength = outputLayer.neuronAmount;

    nn.layers.push_back(std::make_unique<Layer>(hiddenLayer));
    nn.layers.push_back(std::make_unique<Layer>(hiddenLayer2));
    nn.layers.push_back(std::make_unique<Layer>(outputLayer));

    nn.setWeights(inputLength);
    nn.setBiases();

    using clock = std::chrono::high_resolution_clock;

    long long totalNormalizeTime = 0;
    long long totalForwardTime = 0;
    long long totalLossTime = 0;
    long long totalBackwardTime = 0;

    VectorXf inputs(inputLength);
    float loss = 0.0f;
    int correctCount;


    for (int i = 0; i < 1; i++) {
        auto t1 = clock::now();
        normalizeInto(inputs, dataset.training_images[i]);
        auto t2 = clock::now();

        uint8_t correctLabel = dataset.training_labels[i];
        VectorXf correctLabelArr = oneHotEncoding(correctLabel, outputLength);

        auto t3 = clock::now();
        nn.forwardPass(inputs);
        auto t4 = clock::now();

        int predictedLabel = argmax(nn.layers[nn.layers.size()-1]->activations);
        
        if (predictedLabel == correctLabel) {
            correctCount++;
        }

        auto t5 = clock::now();
        loss += lossMSE(correctLabelArr, nn.layers[nn.layers.size()-1]->activations);
        auto t6 = clock::now();

        nn.backwardPass(0.01f, correctLabelArr, inputs);
        auto t7 = clock::now();

        auto normalize_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        auto loss_time = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        auto backward_time = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count();

        // Accumulate times in variables (declared outside the loop)
        totalNormalizeTime += normalize_time;
        totalForwardTime += forward_time;
        totalLossTime += loss_time;
        totalBackwardTime += backward_time;
    }

    std::cout << "Normalization time (total): " << totalNormalizeTime << " us\n";
    std::cout << "Forward pass time (total): " << totalForwardTime << " us\n";
    std::cout << "Loss calculation time (total): " << totalLossTime << " us\n";
    std::cout << "Backward pass time (total): " << totalBackwardTime << " us\n";
}

int main() {
    testPerformance();
    // auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    // std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    // std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    // std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    // std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    // Layer hiddenLayer(256);
    // Layer hiddenLayer2(256);
    // Layer outputLayer(10);

    // Network nn;

    // const size_t inputLength = dataset.training_images[0].size();
    // const size_t outputLength = outputLayer.neuronAmount;

    // nn.layers.push_back(std::make_unique<Layer>(hiddenLayer));
    // nn.layers.push_back(std::make_unique<Layer>(hiddenLayer2));
    // nn.layers.push_back(std::make_unique<Layer>(outputLayer));

    // nn.setWeights(inputLength);
    // nn.setBiases();

    // VectorXf inputs(inputLength);
    // float loss = 0.0f;
    // int correctCount;

    // for (int e = 0; e < 1; e++) {
    //     int correctCount = 0;
    //     float loss = 0.0f;
    //     auto start = std::chrono::high_resolution_clock::now();

    //     for (int i = 0; i < dataset.training_images.size(); i++) {
    //         normalizeInto(inputs, dataset.training_images[i]);
            
    //         uint8_t correctLabel = dataset.training_labels[i];
    //         VectorXf correctLabelArr = oneHotEncoding(correctLabel, outputLength);
            
    //         nn.forwardPass(inputs);
            
    //         int predictedLabel = argmax(nn.layers[nn.layers.size()-1]->activations);
    //         if (predictedLabel == correctLabel) {
    //             correctCount++;
    //         }

    //         loss += lossMSE(correctLabelArr, nn.layers[nn.layers.size()-1]->activations);
    //         nn.backwardPass(0.01f, correctLabelArr, inputs);
    //     }

    //     std::cout << "Loss: " <<loss/dataset.training_images.size() << "\n";
    //     float accuracy = static_cast<float>(correctCount) / dataset.training_images.size();
    //     std::cout << "Epoch " << e + 1 << " Accuracy: " << accuracy * 100.0f << "%\n";
    //     // std::cout << "Loss: " <<loss/10000 << "\n";
    //     // float accuracy = static_cast<float>(correctCount) / 10000;
    //     // std::cout << "Epoch " << e + 1 << " Accuracy: " << accuracy * 100.0f << "%\n";
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    //     std::cout << "Time taken: " << duration.count() << " seconds" << "\n";
    // }

    // std::cout << "Ended" << "\n";
    return 0;
}