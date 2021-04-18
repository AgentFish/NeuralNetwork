#pragma once

/*
FileName: CostFunction
Description: Abstract class for cost functions
Notes: 
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include "../libs/Eigen/Core"

class CostFunction {
private:
    const std::string name; // Function name

public:
    /*
    Name: CostFunction
    Input: name - Cost function name
    Purpose: Class constructor
    Notes: 
    */
    CostFunction(const std::string& name) : name(name) { };

    /*
    Name: ~CostFunction
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~CostFunction(void) = default;

    /*
    Name: getName
    Input: None
    Output: name - Cost function name
    Purpose: Returns the cost function name
    Notes: 
    */
    std::string getName(void) const { return this->name; }

    /*
    Name: calculate
    Input: x - Actual outcome
		   t - Target (expected) outcome
    Output: val - Cost function output
    Purpose: Calculates the cost function
    Notes: 
    */
    virtual double calculate(const Eigen::VectorXd& x, const Eigen::VectorXd& t) const = 0;

    /*
    Name: calculate_derivative
    Input: x - Actual outcome
		   t - Target (expected) outcome
    Output: val - Cost function derivative
    Purpose: Calculates the cost function derivative
    Notes: 
    */
	virtual Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& t) const = 0;
};