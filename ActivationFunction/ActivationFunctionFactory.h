#pragma once

/*
FileName: ActivationFunctionFactory
Description: Static activation function factory
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "Logistic.h"
#include "Softmax.h"

class ActivationFunctionFactory {
public:
    enum class ActivationFunctions {
        LOGISTIC, SOFTMAX
    };

public:
    /*
    Name: create
    Input: name - Activation function enum
    Output: activationFunction - Activation function class
    Purpose: Creates activation function from class enum
    Notes: 
    */
    static std::shared_ptr<ActivationFunction> create(const ActivationFunctions name) {
        switch (name) {
        case ActivationFunctions::LOGISTIC: {
            return std::make_shared<Logistic>();
        }

        case ActivationFunctions::SOFTMAX: {
            return std::make_shared<Softmax>();
        }

        default: {
            throw std::logic_error("ActivationFunctionFactory::create : unimplemented activation function");
        }
        }
    }

    /*
    Name: str2enum
    Input: name - Activation function name
    Output: name - Activation function enum
    Purpose: Returns the enum of the input name
    Notes: 
    */
    static ActivationFunctions str2enum(const std::string& name) {
        if (name.compare(Logistic::Name) == 0) {
            return ActivationFunctions::LOGISTIC;
        }
        else if (name.compare(Softmax::Name) == 0) {
            return ActivationFunctions::SOFTMAX;
        }
        else {
            throw std::logic_error("ActivationFunctionFactory::str2enum : unknown activation function name " + name);
        }
    }
};