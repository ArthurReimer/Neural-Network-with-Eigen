#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
using namespace std;

#include "external/eigen/Eigen/Dense"
using Eigen::MatrixXf;
using Eigen::VectorXf;

typedef Eigen::MatrixXf mat;

namespace act {
    /*
    Sigmoid activation function
    */
    mat sigmoid(mat netInputs) {
        return 1.0f / (1.0f + (-netInputs).array().exp());
    }
    mat derivSigmoid(mat activations) {
        return activations.array() * (1.0f - activations.array());
    }
}


#endif