#pragma once

/*
FileName: NetworkBuilder
Description: Network builder
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include <filesystem>
#include <iostream>
#include <fstream>

#include "Network.h"
#include "../Optimizer/OptimizerFactory.h"
#include "../CostFunction/CostFunctionFactory.h"
#include "../ActivationFunction/ActivationFunctionFactory.h"


class NetworkBuilder {

private:
    size_t inputSize; // Input size
    CostFunctionFactory::CostFunctions costFunction; // Network cost function enum
    OptimizerFactory::Optimizers optimizer; // Network Optimizer enum
    bool isTrueRandom; // Whether or not the seed is truly random

public:
    /*
    Name: NetworkBuilder
    Input: None
    Purpose: Class constructor
    Notes: Note that the first layer is assumed to be an input layer, and by convention we won't set any 
            biases for those neurons, since biases are only ever used in computing the outputs from later layers.
    */
    NetworkBuilder(void) = default;

    /*
    Name: ~NetworkBuilder
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    virtual ~NetworkBuilder(void) = default;

    /*
    Name: setInputSize
    Input: inputSize - Input size
    Output: networkBuilder - reference to this class
    Purpose: Sets the network's input size
    Notes: 
    */
    NetworkBuilder& setInputSize(const size_t inputSize) {
        this->inputSize = inputSize;
        return *this;
    }

    /*
    Name: setCostFunction
    Input: costFunction - Network cost function
    Output: networkBuilder - reference to this class
    Purpose: Sets the network's cost function
    Notes: 
    */
    NetworkBuilder& setCostFunction(const CostFunctionFactory::CostFunctions name) {
        this->costFunction = name;
        return *this;
    }

    NetworkBuilder& setCostFunction(const std::string& name) {
        this->costFunction = CostFunctionFactory::str2enum(name);
        return *this;
    }

    /*
    Name: setOptimizer
    Input: optimizer - Network optimizer
    Output: networkBuilder - reference to this class
    Purpose: Sets the network's optimizer
    Notes: 
    */
    NetworkBuilder& setOptimizer(const OptimizerFactory::Optimizers name) {
        this->optimizer = name;
        return *this;
    }

    NetworkBuilder& setOptimizer(const std::string& name) {
        this->optimizer = OptimizerFactory::str2enum(name);
        return *this;
    }

    /*
    Name: setIsTrueRandom
    Input: isTrueRandom - Whether or not the seed is truly random
    Output: networkBuilder - reference to this class
    Purpose: Sets whether or not the RNG seed is truly random
    Notes: 
    */
    NetworkBuilder& setIsTrueRandom(const bool isTrueRandom) {
        this->isTrueRandom = isTrueRandom;
        return *this;
    }

    /*
    Name: build
    Input: None
    Output: network - Neural network
    Purpose: Builds and returns an empty neural network (no layers within)
    Notes: 
    */
    template <typename PREDICTION>
    std::shared_ptr<Network<PREDICTION>> build(void) const {
        return std::make_shared<Network<PREDICTION>>(this->inputSize,
                                    CostFunctionFactory::create(this->costFunction),
                                    OptimizerFactory::create(this->optimizer),
                                    this->isTrueRandom);
    }

    /*
    Name: createLayer
    Input: size - Number of neurons within the layer
           activationFunction - Layer's activation function enum
    Output: layer - Neural network layer
    Purpose: Builds and returns the a single neural network layer
    Notes: 
    */
    static Layer createLayer(const size_t size, const ActivationFunctionFactory::ActivationFunctions activation) {
        return Layer(size, ActivationFunctionFactory::create(activation));
    }

    /*
    Name: save
    Input: network - Neural network
           filename - Filename to write the network parameters into
    Output: None
    Purpose: Saves the network parameters to a file
    Notes: 
    */
    template <typename PREDICTION>
    static void save(const std::shared_ptr<Network<PREDICTION>>& network, const std::filesystem::path& filename) {
        // Create output format
        // Delimiters are commads ',' (no space)
        // Each line represents either a bias vector or a weight matrix
        // More information at: https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
        Eigen::IOFormat outputFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",", "", "", "", "\n");

        // Open file
        std::ofstream file(filename);
        if (file.is_open()) {
            // Initialize
            Eigen::VectorXd bias;
            Eigen::MatrixXd weight;
            std::shared_ptr<ActivationFunction> activation;

            // Network's input size & cost function
            file << network->inputSize << "," << network->costFunction->getName() << "\n";

            // Run on each layer
            for (auto&& layer : network->layers) {
                // Extract parameters
                layer.getParameters(bias, weight, activation);
                // Save to file
                // 1st = Bias vector
                // 2nd = Weight matrix
                // 3rd = Activation function
                file << bias.format(outputFormat)
                    << weight.format(outputFormat)
                    << activation->getName() << "\n";
            }
            
            // File is closed by the destructor
            // file.close();
        }
        else {
            throw std::runtime_error("NetworkBuilder::save : unable to open file " 
                + filename.u8string() + " for writing network parameters");
        }
    }

    /*
    Name: load
    Input: filename - Filename to load the network parameters from
    Output: None
    Purpose: Loads the network parameters from a file and creates the network
    Notes:
    */
    template <typename PREDICTION>
    std::shared_ptr<Network<PREDICTION>> load(const std::filesystem::path& filename) {
        // Create data stream & open the file
        std::ifstream data;
        data.open(filename);

        if (!data.is_open()) {
            throw std::runtime_error("NetworkBuilder::load : unable to open file " 
                + filename.u8string() + " for loading network parameters");
        }

        // Initialize
        Eigen::VectorXd bias;
        Eigen::MatrixXd weight;
        std::vector<Layer> layers;

        // Initialize for reading
        std::string line;
        std::vector<double> values;
        values.reserve(this->inputSize);
        size_t iRow = 0; // row index

        // Read 1 line at a time
        while (std::getline(data, line)) {
            std::stringstream lineStream(line);
            std::string cell;

            if (iRow > 0) { // Layer parameters
                // 1st = Bias vector
                // 2nd = Weight matrix
                // 3rd = Activation function

                if ((iRow-1) % 3 == 2) { // third row is the activation function
                    // Get activation
                    std::getline(lineStream, cell, ',');
                    // Assign the layer (we already have bias & weight parameters)
                    layers.push_back(NetworkBuilder::createLayer(bias.size(), ActivationFunctionFactory::str2enum(cell)));
                    layers.back().initialize(bias, weight);
                }
                else { 
                    // Push each cell within a line into the values vector
                    while (std::getline(lineStream, cell, ',')) {
                        values.push_back(std::stod(cell));
                    }
                    // Get layer bias or weight
                    if ((iRow-1) % 3 == 0) { // first row is the bias vector
                        bias = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(values.data(), values.size());
                    }
                    else { // second row is the weight matrix
                        weight = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(values.data(), values.size()/bias.rows(), bias.rows()).transpose();
                    }
                }
            }
            else { // First row = network's global parameters
                // 1st  = input size
                std::getline(lineStream, cell, ',');
                this->setInputSize(std::stoull(cell));
                // 2nd = cost function
                std::getline(lineStream, cell, ',');
                this->setCostFunction(cell);
            }

            // Prepare for next iteration
            values.clear();
            ++iRow;
        }

        // Finished - create the network
        auto network = this->build<PREDICTION>();
        for (auto&& layer : layers) {
            network->addLayer(layer, false);
        }

        return network;
    }
};