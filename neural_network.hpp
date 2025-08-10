#pragma once

// Std library
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
#include <string>

// MNIST reader + MNIST data
#include "mnist/include/mnist/mnist_reader.hpp"
using namespace std;

// Eigen
#include "external/eigen/Eigen/Dense"
using Eigen::MatrixXf;
using Eigen::VectorXf;

// Own activations header sccript
#include "activations.hpp"
using namespace act;



#ifndef EIGEN_CORE_H

#else
namespace NN {
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
            MatrixXf derivActivations;
            MatrixXf weightedDeltas;
            act::actFunction activationFunction;
            

            Layer(int n, int batchSize, act::actFunction actFunc) : neuronAmount(n),
                biases(VectorXf::Zero(n)),
                activations(MatrixXf::Zero(n, batchSize)),
                netInputs(MatrixXf::Zero(n, batchSize)),
                deltas(MatrixXf::Zero(n, batchSize)),
                derivActivations(MatrixXf::Zero(n, batchSize)),
                weightedDeltas(MatrixXf::Zero(n, batchSize)),
                activationFunction(actFunc){}
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
            void addLayer(const int n, const act::actFunction actFunc) {
                this->layers.push_back(std::make_unique<Layer>(n, this->batchSize, actFunc));
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
                int layerAmount = this->layers.size();
                std::random_device rd;
                std::mt19937 gen(rd());

                for (int l = 0; l < layerAmount; l++) {
                    Layer& currentLayer = *(this->layers[l]);
                    
                    int cols = (l == 0) ? inputLength : layers[l-1]->neuronAmount;
                    int rows = currentLayer.neuronAmount;

                    currentLayer.weights = MatrixXf(rows, cols);
                    

                    switch (currentLayer.activationFunction) {
                        case RELU: {
                            std::normal_distribution<float> dist(0.0, std::sqrt(2.0 / cols));
                            std::cout << "Using relu weight distribution for layer " << l << endl;

                            for (int r = 0; r < rows; ++r) {
                                for (int c = 0; c < cols; ++c) {
                                    currentLayer.weights(r, c) = dist(gen);
                                } 
                            }
                            break;
                        }
                        case SIGMOID: {
                            std::uniform_real_distribution<> dist(-0.5, 0.5);
                            std::cout << "Using sigmoid weight distribution for layer " << l << endl;

                            for (int r = 0; r < rows; ++r) {
                                for (int c = 0; c < cols; ++c) {
                                    currentLayer.weights(r, c) = dist(gen);
                                } 
                            }
                            break;
                        }
                        case LEAKY_RELU: {
                            std::normal_distribution<float> dist(0.0, std::sqrt(2.0 / cols));
                            std::cout << "Using leaky relu weight distribution for layer " << l << endl;

                            for (int r = 0; r < rows; ++r) {
                                for (int c = 0; c < cols; ++c) {
                                    currentLayer.weights(r, c) = dist(gen);
                                } 
                            }
                            break;
                        }
                        default:
                            std::cout << "Invalid activation function given for layer " << l << "." << endl;
                            exit(0);
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


                for (int l = 0; l < layerAmount; l++) {
                    Layer& currentLayer = *(this->layers[l]);
                    std::vector<float> biases(currentLayer.neuronAmount);

                    std::uniform_real_distribution<> dist;

                    switch (currentLayer.activationFunction) {
                        case RELU:
                            dist = std::uniform_real_distribution<>(0.0, 0.5);
                            std::cout << "Using relu bias distribution for layer " << l  << endl;
                            break;
                        case SIGMOID:
                            dist = std::uniform_real_distribution<>(-0.5, 0.5);
                            std::cout << "Using sigmoid bias distribution for layer " << l  << endl;
                            break;
                        case LEAKY_RELU:
                            dist = std::uniform_real_distribution<>(0.0, 0.5);
                            std::cout << "Using leaky relu bias distribution for layer " << l  << endl;
                            break;
                        default:
                            std::cout << "Invalid activation function given for layer " << l << "." << endl;
                            exit(0);
                    }

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
                    Layer& currentLayer = *(this->layers[l]);
                    int neuronAmount = currentLayer.neuronAmount;

                    /*
                    Calcuation of the net inputs for the current layer
                    Inorder for the dimensions to align with the weighted inputs and the biases, column wise addition is required (colwise())
                    */
                    currentLayer.netInputs = currentLayer.weights * currentInputs;
                    currentLayer.netInputs.colwise() += currentLayer.biases;

                    /*
                    Activation calculation
                    */
                    switch (currentLayer.activationFunction) {
                        case RELU :
                            currentLayer.activations = act::relu(currentLayer.netInputs);
                            break;
                        case SIGMOID:
                            currentLayer.activations = act::sigmoid(currentLayer.netInputs);
                            break;
                        case LEAKY_RELU:
                            currentLayer.activations = act::leakyRelu(currentLayer.netInputs, 0.01);
                            break;
                        default:
                            std::cout << "Invalid activation function given for layer " << l << "." << endl;
                            exit(0);
                    }
                    
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
                int layerAmount = this->layers.size();

                for (int l = layerAmount - 1; l >= 0; l--) {
                    Layer& currentLayer = *(this->layers[l]);
                    int neuronAmount = currentLayer.neuronAmount;

                    /*
                    Calculating all the derivative activations for the current layer
                    */
                    switch (currentLayer.activationFunction) {
                        case RELU:
                            currentLayer.derivActivations = act::derivRelu(currentLayer.netInputs);
                            break;
                        case SIGMOID:
                            currentLayer.derivActivations = act::derivSigmoid(currentLayer.activations);
                            break;
                        case LEAKY_RELU:
                            currentLayer.derivActivations = act::derivLeakyRelu(currentLayer.netInputs, 0.01);
                            break;
                        default:
                            std::cout << "Invalid activation function given for layer " << l << "." << endl;
                            exit(0);
                    }
                    
                    
                    if (l == layerAmount-1) {
                        /*
                        Output layer delta (gradient) calculation
                        .array() has to be used for element wise calculation
                        */
                        currentLayer.deltas = currentLayer.derivActivations.array() * (target.array() - currentLayer.activations.array());
                    } else {
                        /*
                        Hidden layer delta (gradient) calculation
                        Dereferencing the next layer smart pointer
                        .array() has to be used for element wise calculation
                        "nextLayer" is the layer right to the current layer
                        */
                        Layer& nextLayer = *(this->layers[l+1]);
                        
                        currentLayer.weightedDeltas = nextLayer.weights.transpose() * nextLayer.deltas;
                        currentLayer.deltas = currentLayer.derivActivations.array() * currentLayer.weightedDeltas.array();
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
}


#endif // EIGEN_CORE_H