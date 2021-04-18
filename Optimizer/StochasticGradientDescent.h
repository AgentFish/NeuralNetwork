#pragma once

/*
FileName: StochasticGradientDescent
Description: Stochastic gradient descent optimizer
Notes: 
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include "Optimizer.h"

class StochasticGradientDescent : public Optimizer {
public:
    inline static const std::string Name = "stochastic";

public:
    /*
    Name: StochasticGradientDescent
    Input: None
    Purpose: Class constructor
    Notes: 
    */
    StochasticGradientDescent(void) : Optimizer(StochasticGradientDescent::Name) { };

    /*
    Name: ~StochasticGradientDescent
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~StochasticGradientDescent(void) = default;

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
    void optimize(DataLabel_Set& training, const size_t nBatches, const size_t batchSize,
                const double learningRateRatio, const double regularizationRatio) final {

        // Randomly shuffle examples in training set
        std::shuffle(training.begin(), training.end(), this->rngEngine);

        // Initialize
        DataLabel_Set batch(batchSize); // batch
        size_t initialBatchIndex; // initial batch index (within the training set)

        // For each batch
        for (size_t iBatch = 0; iBatch < nBatches; ++iBatch) {
            initialBatchIndex = iBatch * batchSize;
            // Create training batches (swap references, no copy)
            std::swap_ranges(training.begin() + initialBatchIndex, 
                            training.begin() + initialBatchIndex + batchSize,
                            batch.begin());

            // Update network according to the current batch
            this->update_network(batch, learningRateRatio, regularizationRatio);

            // Re-swap to keep the training object full
            std::swap_ranges(batch.begin(),
                            batch.end(),
                            training.begin() + initialBatchIndex);
        }
    };
};