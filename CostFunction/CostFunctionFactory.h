#pragma once

/*
FileName: CostFunctionFactory
Description: Static cost function factory
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "CrossEntropy.h"
#include "QuadraticCostFunction.h"

class CostFunctionFactory {
public:
    enum class CostFunctions {
        QUADRATIC, CROSSENTROPY
    };

public:
    /*
    Name: create
    Input: name - Cost function enum
    Output: costFunction - Cost function class
    Purpose: Creates cost function from class enum
    Notes: 
    */
    static std::shared_ptr<CostFunction> create(const CostFunctions name) {
        switch (name) {
        case CostFunctions::QUADRATIC: {
            return std::make_shared<QuadraticCostFunction>();
        }

        case CostFunctions::CROSSENTROPY: {
            return std::make_shared<CrossEntropy>();
        }

        default: {
            throw std::logic_error("CostFunctionFactory::create : unimplemented cost function");
        }
        }
    }

    /*
    Name: str2enum
    Input: name - Cost function name
    Output: name - Cost function enum
    Purpose: Returns the enum of the input name
    Notes: 
    */
    static CostFunctions str2enum(const std::string& name) {
        if (name.compare(QuadraticCostFunction::Name) == 0) {
            return CostFunctions::QUADRATIC;
        }
        else if (name.compare(CrossEntropy::Name) == 0) {
            return CostFunctions::CROSSENTROPY;
        }
        else {
            throw std::logic_error("CostFunctionFactory::str2enum : unknown cost function name " + name);
        }
    }
};