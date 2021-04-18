#pragma once

/*
FileName: Logistic
Description: Logistic (sigmoid) activation function
Notes: 
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include "ActivationFunction.h"

class Logistic : public ActivationFunction {
public:
    inline static const std::string Name = "logistic";

public:
    /*
    Name: Logistic
    Input: None
    Purpose: Class constructor
    Notes: 
    */
    Logistic(void) : ActivationFunction(Logistic::Name) { };

    /*
    Name: ~Logistic
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~Logistic(void) = default;

    /*
    Name: calculate
    Input: z - Weighted input
    Output: val - Logistic function output
    Purpose: Calculates the logistic function
    Notes: 
    */
    Eigen::VectorXd calculate(const Eigen::VectorXd& z) const final {
        return ((-z.array()).exp() + 1).cwiseInverse(); // 1/(1+exp(-z))
    }

	/*
    Name: calculate_derivative
    Input: z - Weighted input
    Output: val - Logistic function derivative
    Purpose: Calculates the logistic function derivative
    Notes: 
    */
	Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& z) const final {
		Eigen::VectorXd f = this->calculate(z);
		return f.array().cwiseProduct((1 - f.array())); // f(z)*(1-f(z))
	}
};