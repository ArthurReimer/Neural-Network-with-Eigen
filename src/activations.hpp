#pragma once

#include <iostream>
using namespace std;

#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;

typedef Eigen::MatrixXf mat;
typedef Eigen::VectorXf vec;


/*
All activations functions for the network
*/
namespace Activations {
    enum activationFunction {
        RELU,
        SIGMOID,
        LEAKY_RELU,
    };

    /*
    Sigmoid activation function
    */
    inline mat sigmoid(const mat netInputs) {
        return 1.0f / (1.0f + (-netInputs).array().exp());
    }

    /*
    Sigmoid derivative activation function
    */
    inline mat derivSigmoid(const mat activations) {
        return activations.array() * (1.0f - activations.array());
    }

    /*
    Sigmoid activation function
    */
    inline vec sigmoidVec(const vec netInputs) {
        return 1.0f / (1.0f + (-netInputs).array().exp());
    }

    /*
    Sigmoid derivative activation function
    */
    inline vec derivSigmoidVec(const vec activations) {
        return activations.array() * (1.0f - activations.array());
    }

    /*
    Relu activation function
    Goes through all values and sets negatives to zeros
    */
    inline mat relu(const mat netInputs) {
        return netInputs.cwiseMax(0.0f);
    }
    
    /*
    Relu derivative activation function
    */
    inline mat derivRelu(const mat netInputs) {
        return (netInputs.array() > 0).cast<float>();
    }

    /*
    Relu activation function
    Goes through all values and sets negatives to zeros
    */
    inline vec reluVec(const vec netInputs) {
        return netInputs.cwiseMax(0.0f);
    }
    
    /*
    Relu derivative activation function
    */
    inline vec derivReluVec(const vec netInputs) {
        return (netInputs.array() > 0).cast<float>();
    }

    /*
    Leaky relu activation function
    It's similiar to relu, but it allows a small negative value, which helps to prevent neurons from 'dying'
    */
    inline mat leakyRelu(const mat netInputs, const float alpha) {
        return netInputs.cwiseMax(alpha * netInputs);
    }

    /*
    Derivative leaky relu activation function
    */
    inline mat derivLeakyRelu(const mat netInputs, const float alpha) {
        return netInputs.unaryExpr([alpha](float x) {
            return (x > 0) ? 1.0f : alpha;
        });
    }

        /*
    Leaky relu activation function
    It's similiar to relu, but it allows a small negative value, which helps to prevent neurons from 'dying'
    */
    inline vec leakyReluVec(const vec netInputs, const float alpha) {
        return netInputs.cwiseMax(alpha * netInputs);
    }

    /*
    Derivative leaky relu activation function
    */
    inline vec derivLeakyReluVec(const vec netInputs, const float alpha) {
        return netInputs.unaryExpr([alpha](float x) {
            return (x > 0) ? 1.0f : alpha;
        });
    }
}