#pragma once

/*
FileName: ActivationFunction
Description: Abstract class for activation functions
Notes: 
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include "../libs/Eigen/Core"

class ActivationFunction {
private:
    const std::string name; // Function name

public:
    /*
    Name: ActivationFunction
    Input: name - Activation function name
    Purpose: Class constructor
    Notes: 
    */
    ActivationFunction(const std::string& name) : name(name) { };

    /*
    Name: ~ActivationFunction
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~ActivationFunction(void) = default;

    /*
    Name: getName
    Input: None
    Output: name - Activation function name
    Purpose: Returns the activation function name
    Notes: 
    */
    std::string getName(void) const { return this->name; }

    /*
    Name: calculate
    Input: z - Weighted input
    Output: val - Activation function output
    Purpose: Calculates the activation function
    Notes: 
    */
    virtual Eigen::VectorXd calculate(const Eigen::VectorXd& z) const = 0;
    /*
    Name: calculate_derivative
    Input: z - Weighted input
    Output: val - Activation function derivative
    Purpose: Calculates the activation function derivative
    Notes: 
    */
	virtual Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& z) const = 0;
};