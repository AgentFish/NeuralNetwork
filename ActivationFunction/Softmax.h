#pragma once

/*
FileName: Softmax
Description: Softmax activation function
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "ActivationFunction.h"

class Softmax : public ActivationFunction {
public:
    inline static const std::string Name = "softmax";

public:
    /*
    Name: Softmax
    Input: None
    Purpose: Class constructor
    Notes: 
    */
    Softmax(void) : ActivationFunction(Softmax::Name) { };

    /*
    Name: ~Softmax
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~Softmax(void) = default;

    /*
    Name: calculate
    Input: z - Weighted input
    Output: val - Softmax function output
    Purpose: Calculates the softmax function
    Notes: 
    */
    Eigen::VectorXd calculate(const Eigen::VectorXd& z) const final {
        Eigen::VectorXd zExp = z.array().exp(); // TODO imperfect implementation - won't work for large numbers
        return zExp / zExp.sum();
    };

	/*
    Name: calculate_derivative
    Input: z - Weighted input
    Output: val - Softmax function derivative
    Purpose: Calculates the softmax function derivative
    Notes: 
    */
	Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& z) const final {
		throw std::logic_error("Softmax::calculate_derivative : unimplemented method"); // TODO implement this
        // Is this a matrix?
	};
};