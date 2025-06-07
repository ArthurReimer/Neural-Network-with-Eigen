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
    /*
    Sigmoid activation function
    */
    mat sigmoid(mat netInputs) {
        return 1.0f / (1.0f + (-netInputs).array().exp());
    }

    /*
    Sigmoid derivative activation function
    */
    mat derivSigmoid(mat activations) {
        return activations.array() * (1.0f - activations.array());
    }

    /*
    Relu activation function
    Goes through all values and sets negatives to zeros
    */
    mat relu(mat netInputs) {
        return netInputs.cwiseMax(0);
    }

    /*
    Relu derivative activation function
    Converts boolean value to float with .cast
    */
    mat derivRelu(mat activations) {
        return (activations.array() > 0).cast<float>();
    }
}


#endif