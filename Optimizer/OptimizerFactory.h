#pragma once

/*
FileName: OptimizerFactory
Description: Static optimizer factory
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include <stdexcept>

#include "StochasticGradientDescent.h"

class OptimizerFactory {
public:
    enum class Optimizers {
        SGD
    };

public:
    /*
    Name: create
    Input: name - Optimizer enum
    Output: optimizer - Optimizer class
    Purpose: Creates optimizer from class enum
    Notes: 
    */
    static std::shared_ptr<Optimizer> create(const Optimizers name) {
        switch (name) {
        case Optimizers::SGD: {
            return std::make_shared<StochasticGradientDescent>();
        }

        default: {
            throw std::logic_error("OptimizerFactory::create : unimplemented optimizer");
        }
        }
    }

    /*
    Name: str2enum
    Input: name - Optimizer name
    Output: name - Optimizer enum
    Purpose: Returns the enum of the input name
    Notes: 
    */
    static Optimizers str2enum(const std::string& name) {
        if (name.compare(StochasticGradientDescent::Name) == 0) {
            return Optimizers::SGD;
        }
        else {
            throw std::logic_error("OptimizerFactory::str2enum : unknown optimizer name " + name);
        }
    }
};