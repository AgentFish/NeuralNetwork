#pragma once

/*
FileName: CrossEntropy
Description: Cross entropy cost function
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "CostFunction.h"

class CrossEntropy : public CostFunction {
public:
    inline static const std::string Name = "crossentropy";

public:
    /*
    Name: CrossEntropy
    Input: None
    Purpose: Class constructor
    Notes: 
    */
    CrossEntropy(void) : CostFunction(CrossEntropy::Name) { };

    /*
    Name: ~CrossEntropy
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~CrossEntropy(void) = default;

    /*
    Name: calculate
    Input: x - Actual outcome
		   t - Target (expected) outcome
    Output: val - Cross entropy cost function output
    Purpose: Calculates the cross entropy cost function
    Notes: 
    */
    double calculate(const Eigen::VectorXd& x, const Eigen::VectorXd& t) const final {
        // -( t*ln(x) + (1-t)*ln(1-x) )
        Eigen::VectorXd res = -t.array().cwiseProduct(x.array().log()) - (1-t.array()).cwiseProduct((1-x.array()).log());
        return res.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; }).sum();
	};

	/*
    Name: calculate_derivative
    Input: x - Actual outcome
		   t - Target (expected) outcome
    Output: val - Cross entropy cost function derivative
    Purpose: Calculates the cross entropy cost function derivative
    Notes: 
    */
	Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& t) const final {
        // ( x-t ) / ( x*(1-x) )
		return (x - t).array().cwiseQuotient(x.array().cwiseProduct(1-x.array()));
	};
};