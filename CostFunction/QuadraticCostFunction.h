#pragma once

/*
FileName: QuadraticCostFunction
Description: Quadratic cost function
Notes: 
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include "CostFunction.h"

class QuadraticCostFunction : public CostFunction {
public:
    inline static const std::string Name = "quadratic";

public:
    /*
    Name: QuadraticCostFunction
    Input: None
    Purpose: Class constructor
    Notes: 
    */
    QuadraticCostFunction(void) : CostFunction(QuadraticCostFunction::Name) { };

    /*
    Name: ~QuadraticCostFunction
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~QuadraticCostFunction(void) = default;

    /*
    Name: calculate
    Input: x - Actual outcome
		   t - Target (expected) outcome
    Output: val - Quadratic cost function output
    Purpose: Calculates the quadratic cost function
    Notes: 
    */
    double calculate(const Eigen::VectorXd& x, const Eigen::VectorXd& t) const final {
        // 0.5*(t - x)^2
        return 0.5*pow((t - x).norm(), 2);
	};

	/*
    Name: calculate_derivative
    Input: x - Actual outcome
		   t - Target (expected) outcome
    Output: val - Quadratic cost function derivative
    Purpose: Calculates the quadratic cost function derivative
    Notes: 
    */
	Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& t) const final {
        // x- t
		return (x - t);
	};
};