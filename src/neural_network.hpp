#pragma once


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


#include "activations.hpp"
using namespace Activations;

#pragma GCC push_options
#pragma GCC optimize("O0")
#include "mnist/mnist_reader.hpp"
#pragma GCC pop_options

#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;


// type defs
using WeightMatrix = Eigen::MatrixXf;
using BiasVector = Eigen::VectorXf;
using ActivationVector = Eigen::VectorXf;
using NetinputVector = Eigen::VectorXf;
using GradientVector = Eigen::VectorXf;
using InputVector = Eigen::VectorXf;
using Activation = Activations::activationFunction;


void printMatrix(Eigen::MatrixXf mat) {
    std::cout << "{" << "\n";
    for (int row = 0; row < mat.rows(); row++) {
        std::cout << "{";
        for (int col = 0; col < mat.cols(); col++) {
            std::cout << mat(row, col) << " ";
        }
        std::cout << "}" << "\n";
    }
    std::cout << "}" << "\n";
}

void printVector(Eigen::VectorXf vec) {
    std::cout << "{";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec(i) << " ";
    }
    std::cout << "}" <<  "\n";
}

class DenseLayer {
    public:
        size_t neuronAmount;
        size_t inputNeuronAmount;
        WeightMatrix weights;
        BiasVector biases;
        ActivationVector activations;
        ActivationVector derivActivations;
        NetinputVector netinputs;
        GradientVector gradients;
        GradientVector weightedGradients;
        Activation activation;


    DenseLayer(size_t inputNeuronAmount, size_t neuronAmount, Activation activationFunc) :
        neuronAmount(neuronAmount),
        inputNeuronAmount(inputNeuronAmount),
        biases(VectorXf::Zero(neuronAmount)),
        activations(ActivationVector::Zero(neuronAmount)),
        derivActivations(ActivationVector::Zero(neuronAmount)),
        netinputs(NetinputVector::Zero(neuronAmount)),
        gradients(GradientVector::Zero(neuronAmount)),
        weightedGradients(GradientVector::Zero(neuronAmount)),
        weights(WeightMatrix::Zero(neuronAmount, inputNeuronAmount)),
        activation(activationFunc)
        {
            this->generateRandomWeights();
            this->generateRandomBiases();
        }


    void generateRandomWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());

        float stddev = (activation == RELU || activation == LEAKY_RELU)
            ? std::sqrt(2.0f / inputNeuronAmount)
            : std::sqrt(2.0f / (inputNeuronAmount + neuronAmount));
        std::normal_distribution<float> dist(0.0f, stddev);

        for (int r = 0; r < neuronAmount; ++r) {
            for (int c = 0; c < inputNeuronAmount; ++c) {
                this->weights(r, c) = dist(gen);
            } 
        }
    }


    void generateRandomBiases() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 0.1f);

        for (int r = 0; r < neuronAmount; ++r) {
            this->biases(r) = dist(gen);
        }
    }
};

class NeuralNetwork {
    public:
        size_t inputLength;
        std::vector<DenseLayer> layers;

    NeuralNetwork(size_t inputLength) 
    : inputLength(inputLength) {
        layers.reserve(10);
    }


    void addLayer(size_t neuronAmount, Activation activationFunc) {
        if (layers.empty()) {
            this->layers.push_back(DenseLayer(this->inputLength, neuronAmount, activationFunc));
        } else {
            size_t ind = layers.size()-1;
            this->layers.push_back(DenseLayer(this->layers[ind].neuronAmount, neuronAmount, activationFunc));
        }
    }


    void displayLayers() {
        for (size_t layerInd = 0; layerInd < this->layers.size(); layerInd++) {
            DenseLayer &layer = this->layers[layerInd];
            std::cout << "Dense layer " << layerInd << "\n";
            std::cout << "- Weight size " << layer.weights.size() << "\n";
            std::cout << "- Input length " << layer.inputNeuronAmount << "\n";
            std::cout << "- Neuron amount " << layer.neuronAmount << "\n";
        }
    }


    void forwardpass(InputVector &inputs, bool debug = false) {
        for (size_t layerInd = 0; layerInd < this->layers.size(); layerInd++) {
            DenseLayer &layer = this->layers[layerInd];
            // activations of the layer left of the current layer "layer"
            ActivationVector& previousActivations = (layerInd == 0) ? inputs : this->layers[layerInd - 1].activations; 
            layer.netinputs.noalias() = layer.weights * previousActivations + layer.biases;
            
            switch (layer.activation) {
                case RELU: {
                    layer.activations = Activations::reluVec(layer.netinputs);
                    break;
                }
                case LEAKY_RELU: {
                    layer.activations = Activations::leakyReluVec(layer.netinputs, 0.001f);
                    break;
                }
                case SIGMOID: {
                    layer.activations = Activations::sigmoidVec(layer.netinputs);
                    break;
                }
                default:
                    std::cout << "Invalid activation function " << layer.activation << " for layer " << layerInd << "." << std::endl;
                    exit(0);
            }

            if (debug) {
                std::cout << "Netinputs";
                printVector(layer.netinputs);
                std::cout << "\n" << "Activations";
                printVector(layer.activations);
            }
        } 
    }


    void backpropagation(InputVector &inputs, double learningRate, Eigen::VectorXf target, bool debug = false) {
        for (int l = this->layers.size() - 1; l >= 0; l--) {
            DenseLayer &layer = this->layers[l];
                
            switch (layer.activation) {
                case RELU: {
                    layer.derivActivations = Activations::derivReluVec(layer.netinputs);
                    break;
                }
                case LEAKY_RELU: {
                    layer.derivActivations = Activations::derivLeakyReluVec(layer.netinputs, 0.001f);
                    break;
                }
                case SIGMOID: {
                    layer.derivActivations = Activations::derivSigmoidVec(layer.activations);
                    break;
                }
                default:
                    std::cout << "Invalid activation function " << layer.activation << " for layer " << l << "." << std::endl;
                    exit(0);
            }

            if (l == this->layers.size() - 1) {
                layer.gradients = layer.derivActivations.array() * (layer.activations.array() - target.array());
            } else {
                DenseLayer &rightLayer = this->layers[l+1];
                layer.gradients = layer.derivActivations.array() * (rightLayer.weights.transpose() * rightLayer.gradients).array();
            }

            VectorXf &activationsLayerLeft = (l == 0) ? inputs : layers[l - 1].activations;
            layer.biases.noalias() -= learningRate * layer.gradients;
            layer.weights.noalias() -= learningRate * layer.gradients * activationsLayerLeft.transpose();
        }
    }

    size_t getWeightAmount() {
        size_t weights = 0;
        for (DenseLayer layer : this->layers) {
            weights += layer.weights.size();
        }
        return weights;
    }

    size_t getBiasAmount() {
        size_t biases = 0;
        for (DenseLayer layer : this->layers) {
            biases += layer.biases.size();
        }
        return biases;
    }
};