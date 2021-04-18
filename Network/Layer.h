#pragma once

/*
FileName: Layer
Description: Single neural network layer
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "../ActivationFunction/ActivationFunction.h"

class Layer {

private:
	size_t size; // Number of neurons
    std::shared_ptr<ActivationFunction> activationFunction; // Activation function

	Eigen::VectorXd bias; // Bias vector values
    Eigen::MatrixXd weight; // Weight matrix values

public:
    /*
    Name: Layer
    Input: size - Number of neurons
           activationFunction - Activation function
    Purpose: Constructs a single neural network layer
    Notes: 
    */
    Layer(const size_t size, std::shared_ptr<ActivationFunction>& activationFunction) {
        // Assign inputs
        this->size = size;
        this->activationFunction = activationFunction;
    }

    /*
    Name: ~Layer
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~Layer(void) = default;

    /*
    Name: initialize
    Input: previousLayerSize - Previous layer size
           generatorFunc - Initial bias & weight generator function
    Output: None
    Purpose: Initializes the layer's bias vector & weight matrix with random parameters
    Notes: 
    */
    void initialize(const size_t previousLayerSize, const std::function<double(double)>& generatorFunc) {
        // Assign initial random bias & weight values
        // Note: the weight is divided by the square root of the number of connections input to the neuron
        this->bias = Eigen::VectorXd::Zero(this->getSize()).unaryExpr(generatorFunc);
        this->weight = Eigen::MatrixXd::Zero(this->getSize(), previousLayerSize).unaryExpr(generatorFunc) / sqrt(previousLayerSize);
    }

    /*
    Name: initialize
    Input: bias - Bias vector values
           weight - Weight matrix values
    Output: None
    Purpose: Initializes the layer's bias vector & weight matrix from input
    Notes: Used when loading parameters from file
    */
    void initialize(const Eigen::VectorXd& bias, const Eigen::MatrixXd& weight) {
        // Assign bias & weight values
        this->bias = bias;
        this->weight = weight;
    }

    /*
    Name: getParameters
    Input: None
    Output: bias - Bias vector values
            weight - Weight matrix values
            activationFunction - Activation function
    Purpose: Returns the layer's bias vector & weight matrix
    Notes: Used for writing parameters into file
    */
    void getParameters(Eigen::VectorXd& bias, Eigen::MatrixXd& weight, std::shared_ptr<ActivationFunction>& activationFunction) const {
        bias = this->bias;
        weight = this->weight;
        activationFunction = this->activationFunction;
    }

    /*
    Name: getSize
    Input: None
    Output: size - Number of neurons
    Purpose: Returns the number of neurons within the layer
    Notes: 
    */
	size_t getSize(void) const {
		return this->size;
	}

    /*
    Name: feedForward
    Input: x - Input data or features
    Output: a - Layer's output (prediction)
    Purpose: Propagates the input to calculate the layer's output
    Notes: 
    */
	Eigen::VectorXd feedForward(const Eigen::VectorXd& x) const {
        return this->activationFunction->calculate(this->weight * x + this->bias);
    }

    /*
    Name: feedForward
    Input: x - Input data or features
    Output: a - Layer's activation (prediction)
            z - Layer's Weighted input
    Purpose: Propagates the input to calculate the weighted input & activation
    Notes: To be used to save the layer's weighted input & activation
    */
	void feedForward(Eigen::VectorXd& a, Eigen::VectorXd& z, const Eigen::VectorXd& x) const {
        z = this->weight * x + this->bias;
        a = this->activationFunction->calculate(z);
    }

    /*
    Name: feedBackward
    Input: deltaPrevious - Upper layer's delta (error)
           a - Lower layer's activation
           z - Layer's weighted input
    Output: delta_nabla_b - Delta difference for bias vector
            delta_nabla_w - Delta difference for weight matrix
            deltaPrevious - Current layer's delta (error)
    Purpose: Calculates the required changes in the nablas according to the delta (error)
    Notes: 
    */
	void feedBackward(Eigen::VectorXd& delta_nabla_b, Eigen::MatrixXd& delta_nabla_w, Eigen::VectorXd& deltaPrevious,
                    const Eigen::VectorXd& a, const Eigen::VectorXd& z) const {

        Eigen::VectorXd sigmaDerivative = this->activationFunction->calculate_derivative(z);
        Eigen::VectorXd delta = deltaPrevious.cwiseProduct(sigmaDerivative);

        // Assign
        delta_nabla_w = delta * a.transpose();
        delta_nabla_b = delta;
        // Update delta for next layer
        deltaPrevious = this->weight.transpose() * delta;
    }

    /*
    Name: updateBiasWeight
    Input: nabla_b - Bias vector difference
           nabla_w - Weight matrix difference
           learningRateRatio - Learning ratio for updating the parameters
           regularizationRatio - Regularization ratio for updating the parameters
    Output: None
    Purpose: Updates the weights & biases according to their differences
    Notes:  Uses the regularization ratio for the weights only
    */
    void updateBiasWeight(Eigen::VectorXd& nabla_b, Eigen::MatrixXd& nabla_w,
                        const double learningRateRatio, const double regularizationRatio) {
        this->bias += learningRateRatio * nabla_b;
        this->weight += learningRateRatio * nabla_w + regularizationRatio * this->weight;
    }
};