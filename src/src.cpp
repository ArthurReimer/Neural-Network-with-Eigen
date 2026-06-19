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
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;

// type defs
using WeightMatrix = Eigen::MatrixXf;
using BiasVector = Eigen::VectorXf;
using ActivationVector = Eigen::VectorXf;
using NetinputVector = Eigen::VectorXf;
using GradientVector = Eigen::VectorXf;

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


    DenseLayer(size_t inputNeuronAmount, size_t neuronAmount, size_t batchSize) :
        neuronAmount(neuronAmount),
        inputNeuronAmount(inputNeuronAmount),
        biases(VectorXf::Zero(neuronAmount)),
        activations(ActivationVector::Zero(neuronAmount)),
        derivActivations(ActivationVector::Zero(neuronAmount)),
        netinputs(NetinputVector::Zero(neuronAmount)),
        gradients(GradientVector::Zero(neuronAmount)),
        weightedGradients(GradientVector::Zero(neuronAmount)),
        weights(WeightMatrix::Zero(neuronAmount, inputNeuronAmount)) 
        {
            this->generateRandomWeights();
            this->generateRandomBiases();
        }


    void generateRandomWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, std::sqrt(2.0 / inputNeuronAmount));

        for (int r = 0; r < neuronAmount; ++r) {
            for (int c = 0; c < inputNeuronAmount; ++c) {
                this->weights(r, c) = dist(gen);
            } 
        }
    }


    void generateRandomBiases() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 1);

        for (int r = 0; r < neuronAmount; ++r) {
            this->biases(r) = dist(gen);
        }
    }
};

class NeuralNetwork {
    public:
        size_t batchSize;
        size_t inputLength;
        std::vector<DenseLayer> layers;

    NeuralNetwork(size_t batchSize, size_t inputLength) 
    : batchSize(batchSize), inputLength(inputLength) {}

    void addLayer(size_t neuronAmount) {


        // layers.push_back()
    }


};






int main() {
    DenseLayer dl = DenseLayer(32, 10, 1);
    printVector(dl.biases);
    return 0;
}