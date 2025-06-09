#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
using namespace std;

#include "external/eigen/Eigen/Dense"
using Eigen::MatrixXf;
using Eigen::VectorXf;

typedef Eigen::MatrixXf mat;


/*
All activations functions for the network
*/
namespace act {
    enum actFunction {
        RELU,
        SIGMOID,
        LEAKY_RELU,
    };

    /*
    Sigmoid activation function
    */
    mat sigmoid(const mat netInputs) {
        return 1.0f / (1.0f + (-netInputs).array().exp());
    }

    /*
    Sigmoid derivative activation function
    */
    mat derivSigmoid(const mat activations) {
        return activations.array() * (1.0f - activations.array());
    }

    /*
    Relu activation function
    Goes through all values and sets negatives to zeros
    */
    mat relu(const mat netInputs) {
        return netInputs.cwiseMax(0.0f);
    }
    
    /*
    Relu derivative activation function
    Converts boolean value to float with .cast
    */
    mat derivRelu(const mat netInputs) {
        return (netInputs.array() > 0).cast<float>();
    }

    /*
    Leaky relu activation function
    It's similiar to relu, but it allows small negative value, which helps to prevent neurons from 'dying'
    */
    mat leakyRelu(const mat netInputs, const float α) {
        return netInputs.cwiseMax(α * netInputs);
    }

    /*
    Derivative leaky relu activation function
    */
    mat derivLeakyRelu(const mat netInputs, const float α) {
        return netInputs.unaryExpr([α](float x) {
            return (x > 0) ? 1.0f : α;
        });
    }
}

#endif