#pragma once

/*
FileName: Network
Description: Fully connected neural network
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "Layer.h"
#include "../CostFunction/CostFunction.h"
#include "../Optimizer/Optimizer.h"

template <typename PREDICTION>
class Network {

public:
    // For printing
    std::vector<double> trainingCost; // Training cost
    std::vector<double> trainingAccuracy; // Training accuracy
    std::vector<double> evaluationCost; // Evaluation cost
    std::vector<double> evaluationAccuracy; // Evaluation accuracy

protected:
    // Type alias
    using DataLabel_Pair = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
    using DataLabel_Set = std::vector<DataLabel_Pair>;

private:
    friend class NetworkBuilder;

    size_t inputSize; // Input size (= first 'layer''s size)
    std::vector<Layer> layers; // Vector of neural network layers

    std::shared_ptr<CostFunction> costFunction; // Cost function
    std::shared_ptr<Optimizer> optimizer; // Training optimizer

    std::default_random_engine rngEngine; // Random Number Generator (RNG) engine
    std::function<double(void)> layerParamsInitializer; // RNG function
    std::function<double (double)> layerEigenParamsInitializer; // RNG function for Eigen parameters

public:
    /*
    Name: Network
    Input: inputSize - Input size
           costFunction - Network cost function
           optimizer - Network optimizer
           isTrueRandom - Whether or not the seed is truly random
    Purpose: Initializes the network bias and weight vectors of each layer
    Notes: Note that the first layer is assumed to be an input layer, and by convention we won't set any biases 
            for those neurons, since biases are only ever used in computing the outputs from later layers.
    */
    Network(const size_t inputSize, std::shared_ptr<CostFunction>& costFunction,
            std::shared_ptr<Optimizer>& optimizer, const bool isTrueRandom) {

        // Assign inputs
        this->inputSize = inputSize;
        this->costFunction = costFunction;
        this->optimizer = optimizer;

        // Initialize Random Number Generator (RNG) engine
        if (isTrueRandom) {
            // Seed with a real random value
            std::random_device r;
            this->rngEngine.seed(r());
        }
        else {
            this->rngEngine.seed(17111993); // pseudo-random number
        }
        // Initialize RNG function
        auto distribution = std::normal_distribution<double>(0, 1);
        // Create a function to be used with Eigen unary expression
        this->layerParamsInitializer = std::bind(distribution, this->rngEngine);
        this->layerEigenParamsInitializer = [&](double){ return this->layerParamsInitializer(); }; // unused input

        // Initialize optimizer
        std::function<void(const DataLabel_Set&, const double, const double)> update_func = std::bind(&Network::updateParameters, 
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        this->optimizer->initialize(this->rngEngine, update_func);
	}

    /*
    Name: ~Network
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~Network(void) = default;

    /*
    Name: getNumberOfLayers
    Input: None
    Output: nLayers - Number of network layers
    Purpose: Returns the number of layers within the network
    Notes: 
    */
	size_t getNumberOfLayers(void) const {
		return this->layers.size();
	}

    /*
    Name: printLayers
    Input: None
    Output: None
    Purpose: Prints number of neurons within each layer
    Notes: 
    */
    void printLayers(void) const {
        if (this->getNumberOfLayers() == 0) { // empty network
            std::cout << "The neural network is empty." << std::endl;
        }
        else { // not empty network
            std::cout << "The neural network has " << this->getNumberOfLayers() << " layers:\n";
            // Input layer
            std::cout << "    Input : " << this->inputSize << " neurons\n";
            // Hidden layers
            for (size_t iLayer = 0; iLayer < this->getNumberOfLayers()-1; ++iLayer) {
                std::cout << "\t" << iLayer << " : " << this->layers[iLayer].getSize() << " neurons\n";
            }
            // Output layer
            std::cout << "   Output : " << this->layers.back().getSize() << " neurons\n\n";
        }
    }

    /*
    Name: addLayer
    Input: layer - Neural network layer
           isNotInitialized - Whether the layer is not initialized
    Output: network - reference to this class
    Purpose: Adds & initializes the next neural network layer
    Notes: 
    */
    Network<PREDICTION>& addLayer(Layer layer, const bool isNotInitialized = true) {
        if (isNotInitialized) {
            // Get previous layer size
            size_t previousLayerSize;
            if (this->getNumberOfLayers() == 0) { // first layer
                previousLayerSize = this->inputSize;
            }
            else {
                previousLayerSize = this->layers.back().getSize();
            }
            // Initialize layer
            layer.initialize(previousLayerSize, this->layerEigenParamsInitializer);
        }
        // Add layer
        this->layers.push_back(layer);

        return *this;
    }

    /*
    Name: train
    Input: training - Training data (Data & Label pairs)
           evaluation - Evaluation data (Data & Label pairs)
           nEpoch - Number of epochs
           batchSize - Batch size
           eta - Learning rate
           lambda - Regularization parameter
    Output: None
    Purpose: Trains (optimizes) the network
    Notes: 
    */
    // TODO use early stopping - stop the training when the best classification accuracy doesn't improve for some epochs
    // TODO use learning rate schedule - Keep lowering the learning rate as time progresses
    // TODO use the momentum-based gradient descent
    void train(DataLabel_Set& training, const DataLabel_Set& evaluation,
        const size_t nEpoch, const size_t batchSize, const double eta, const double lambda) {

        // Check input layer size
        if (training.front().first.size() != this->inputSize) {
            throw std::range_error("Network::train : input layer size ("
                + std::to_string(this->inputSize) + ") is inconsistent with training input data size ("
                + std::to_string(training.front().first.size()) + ")");
        }
        // Check output layer size
        if (training.front().second.size() != this->layers.back().getSize()) {
            throw std::range_error("Network::train : output layer size ("
                + std::to_string(this->layers.back().getSize()) + ") is inconsistent with training output data size ("
                + std::to_string(training.front().second.size()) + ")");
        }

        // Initialize
        size_t nTraining = training.size(); // Number of training cases
        size_t nBatches = nTraining / batchSize; // Number of batches
        size_t nEvaluation = evaluation.size(); // Number of evaluation cases
        // Create learning ratio
        double learningRateRatio = -eta/batchSize; // learning ratio for updating the parameters
        double regularizationRatio = -eta*lambda/nTraining; // regularization ratio for updating the parameters

        // Reserve length
        this->trainingCost.reserve(nEpoch);
        this->trainingAccuracy.reserve(nEpoch);
        this->evaluationCost.reserve(nEpoch);
        this->evaluationAccuracy.reserve(nEpoch);

        // For each training epoch
        for (size_t iEpoch = 0; iEpoch < nEpoch; ++iEpoch) {

            // Train a single epoch
            this->optimizer->optimize(training, nBatches, batchSize, learningRateRatio, regularizationRatio);

            // Evaluate at the end of the epoch
            auto [nTrainingSuccess, trainingCost] = this->calcAccuracyAndCost(training);
            auto [nEvaluationSuccess, evaluationCost] = this->calcAccuracyAndCost(evaluation);
            trainingCost /= nTraining;
            evaluationCost /= nEvaluation;
            std::cout << "Epoch # " << iEpoch << " of training is complete:\n"
                << "\tCost on training data: " << trainingCost << "\n"
                << "\tAccuracy on training data: " << nTrainingSuccess << " / " << nTraining << "\n"
                << "\tCost on evaluation data: " << evaluationCost << "\n"
                << "\tAccuracy on evaluation data: " << nEvaluationSuccess << " / " << nEvaluation << std::endl;
            // Add
            this->trainingCost.push_back(trainingCost);
            this->trainingAccuracy.push_back(static_cast<double>(nTrainingSuccess)/nTraining);
            this->evaluationCost.push_back(evaluationCost);
            this->evaluationAccuracy.push_back(static_cast<double>(nEvaluationSuccess)/nEvaluation);
        }
    }

    /*
    Name: predict
    Input: input - Input data or features
    Output: p - Network's prediction
    Purpose: Returns the network's prediction
    Notes: 
    */
    PREDICTION predict(const Eigen::VectorXd& input) const {
        return this->output2prediction(this->feedForward(input));
    }

    /*
    Name: calcAccuracyAndCost
    Input: data - Evaluation data (Data & Label pairs)
           lambda - Regularization parameter
    Output: correct - Number of correct evaluation cases
            cost - Total cost of the data set (not normalized by data length)
    Purpose: Calculates the network's accuracy & cost for the input data
    Notes: 
    */
	std::tuple<size_t, double> calcAccuracyAndCost(const DataLabel_Set& data, const double lambda = 0) const {
        // Initialize outputs
        size_t correct = 0;
        double cost = 0;

        // Initialize
        Eigen::VectorXd predictedOutput;
        Eigen::VectorXd expectedOutput;
        PREDICTION prediction;
        PREDICTION expected;

        // Run on each data input
        for (auto&& input : data) {
            // Prediction
            predictedOutput = this->feedForward(input.first);
            prediction = this->output2prediction(predictedOutput);
            // Expected
            this->prediction2output(expectedOutput, input.second);
            expected = this->output2prediction(input.second);

            // Accuracy - compare
            if (prediction == expected) {
                correct++;
            }
            // Cost
            cost += this->costFunction->calculate(predictedOutput, expectedOutput);
        }

        // Cost regularization term
        double costRegularization = 0;
        // Initialize
        Eigen::VectorXd bias; // unused
        Eigen::MatrixXd weight;
        std::shared_ptr<ActivationFunction> activationFunction; // unused
        for (auto&& layer : this->layers) {
            layer.getParameters(bias, weight, activationFunction);
            costRegularization += pow(weight.norm(), 2);
        }
        cost += (lambda/2)*costRegularization;

        return {correct, cost};
    }

private:
    /*
    Name: output2prediction
    Input: output - Network's output
    Output: prediction - Network's prediction
    Purpose: Converts the network's output to prediction
    Notes: 
    */
    virtual PREDICTION output2prediction(const Eigen::VectorXd& output) const {
        // Initialize
        PREDICTION prediction;

        // Determine according to size
        if(output.size() > 1) {
            // Vector - choose maximal
            output.maxCoeff(&prediction);
        }
        else {
            // Scalar - choose the only element
            prediction = static_cast<PREDICTION>(output(0));
        }
        return prediction;
    }

    /*
    Name: prediction2output
    Input: prediction - Network's prediction or data's label
    Output: output - Network's output
    Purpose: Converts a prediction to the the network's output
    Notes: 
    */
    virtual void prediction2output(Eigen::VectorXd& output, const Eigen::VectorXd& prediction) const {
        if(prediction.size() > 1) {
            // Vector - take all
            output = prediction;
        }
        else {
            // Scalar - turn into a vector
            output = Eigen::VectorXd::Zero(this->layers.back().getSize());
            output(static_cast<PREDICTION>(prediction(0))) = 1;
        }
    }

    /*
    Name: feedForward
    Input: x - Input data or features
    Output: a - Network's output
    Purpose: Propagates the input to calculate the network's output
    Notes: 
    */
	Eigen::VectorXd feedForward(const Eigen::VectorXd& x) const {
        // Initialize activation
        Eigen::VectorXd a = x;
        // Feed forward (propagate) the inputs of each layer
        for (auto&& layer : this->layers) {
            a = layer.feedForward(a);
        }
        return a;
    }

    /*
    Name: updateParameters
    Input: training - Training data (Data & Label pairs)
           learningRateRatio - Learning ratio for updating the parameters
           regularizationRatio - Regularization ratio for updating the parameters
    Output: None
    Purpose: Updates the layer's weight & bias according to the training data using backpropagation
    Notes: 
    */
    void updateParameters(const DataLabel_Set& training, const double learningRateRatio, const double regularizationRatio) {
        // Initialize nablas
        std::vector<Eigen::VectorXd> nabla_b(this->getNumberOfLayers());
        std::vector<Eigen::MatrixXd> nabla_w(this->getNumberOfLayers());
        // First layer
        size_t iLayer = 0;
        nabla_b[iLayer] = Eigen::VectorXd::Zero(this->layers[iLayer].getSize());
        nabla_w[iLayer] = Eigen::MatrixXd::Zero(this->layers[iLayer].getSize(), this->inputSize);
        // Second layer up to last
        for (++iLayer; iLayer < this->getNumberOfLayers(); ++iLayer) { // note initial ++iLayer
            // Bias vector for each layer
            nabla_b[iLayer] = Eigen::VectorXd::Zero(this->layers[iLayer].getSize());
            // Weight matrix where each row is for next layer, connecting to the previous layer
            nabla_w[iLayer] = Eigen::MatrixXd::Zero(this->layers[iLayer].getSize(), this->layers[iLayer-1].getSize());
        }
        // Initialize delta nablas (same size as the nablas)
        std::vector<Eigen::VectorXd> delta_nabla_b = nabla_b;
        std::vector<Eigen::MatrixXd> delta_nabla_w = nabla_w;

        // Update nabla after each training session
        for (size_t idx = 0; idx <  training.size(); ++idx) {
            this->backPropagate(delta_nabla_b, delta_nabla_w, training[idx].first, training[idx].second);
            for (iLayer = 0; iLayer < this->getNumberOfLayers(); ++iLayer) {
                nabla_b[iLayer] += delta_nabla_b[iLayer];
                nabla_w[iLayer] += delta_nabla_w[iLayer];
            }
        }

        // Update bias & weight
        for (iLayer = 0; iLayer < this->getNumberOfLayers(); ++iLayer) {
            this->layers[iLayer].updateBiasWeight(nabla_b[iLayer], nabla_w[iLayer],
                                                    learningRateRatio, regularizationRatio);
        }
    }

    /*
    Name: backPropagate
    Input: x - Data
           y - Data's label
    Output: delta_nabla_b - Delta nabla for biases
            delta_nabla_w - Delta nabla for weights
    Purpose: Calculates the required changes in the nablas to predict correctly the data's label
    Notes: 
    */
	void backPropagate(std::vector<Eigen::VectorXd>& delta_nabla_b, std::vector<Eigen::MatrixXd>& delta_nabla_w, 
                        const Eigen::VectorXd& x, const Eigen::VectorXd& y) const {
        // TODO modify to accept batch of x & y (matrices), instead of a single input vector
        // Initialize activations
        std::vector<Eigen::VectorXd> zs(this->getNumberOfLayers()); // store all the weighted inputs, layer by layer
        std::vector<Eigen::VectorXd> activations(this->getNumberOfLayers()+1); // store all the activations, layer by layer

        // Feed forward
        // Initialize first layer's activation (which is simply the inputs)
        int iLayer = 0;
        activations[iLayer] = x;
        for (; iLayer < this->getNumberOfLayers(); ++iLayer) {
            this->layers[iLayer].feedForward(activations[iLayer+1], zs[iLayer], activations[iLayer]);
        }

        // Feed backward
        // Initialize last layer's delta (according to cost function & network's output)
        Eigen::VectorXd delta = this->costFunction->calculate_derivative(activations[iLayer], y);
        // Last layer up to first
        for (--iLayer; iLayer >= 0; --iLayer) { // note initial --iLayer
            // keep updating delta
            this->layers[iLayer].feedBackward(delta_nabla_b[iLayer], delta_nabla_w[iLayer], delta, activations[iLayer], zs[iLayer]);
        }
    }
};