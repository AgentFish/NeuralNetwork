#pragma once

/*
FileName: Optimizer
Description: Abstract class for optimizers
Notes: 
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include <vector>
#include <algorithm>
#include <random>
#include <functional>

class Optimizer {
protected:
    // Type alias
    using DataLabel_Pair = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
    using DataLabel_Set = std::vector<DataLabel_Pair>;

    // Network-related parameters
    std::function<void(const DataLabel_Set&, const double, const double)> update_network;
    std::default_random_engine rngEngine; // Random Number Generator (RNG) engine

private:
    const std::string name; // Optimizer name

public:
    /*
    Name: Optimizer
    Input: name - Optimizer name
    Purpose: Class constructor
    Notes: 
    */
    Optimizer(const std::string& name) : name(name) { };

    /*
    Name: ~Optimizer
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~Optimizer(void) = default;

    /*
    Name: getName
    Input: None
    Output: name - Optimizer name
    Purpose: Returns the optimizer name
    Notes: 
    */
    std::string getName(void) const { return this->name; }

    /*
    Name: initialize
    Input: rngEngine - Random Number Generator (RNG) engine
           update_network - Network's weights & biases update function
    Output: None
    Purpose: Initializes the optimizer with the network related functions
    Notes: 
    */
    void initialize(std::default_random_engine& rngEngine,
                    const std::function<void(const DataLabel_Set&, const double, const double)>& update_network) {
        this->rngEngine = rngEngine;
        this->update_network = update_network;
    }

    /*
    Name: optimize
    Input: training - Training data (Data & Label pairs)
           nBatches - Number of batches to run
           batchSize - Batch size
           learningRateRatio - Learning ratio for updating the parameters
           regularizationRatio - Regularization ratio for updating the parameters
    Output: None
    Purpose: Optimizes the network
    Notes: 
    */
    virtual void optimize(DataLabel_Set& training, const size_t nBatches, const size_t batchSize,
                        const double learningRateRatio, const double regularizationRatio) = 0;
};