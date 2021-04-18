#pragma once

/*
FileName: Manager
Description: Neural network manager
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#include "Network/NetworkBuilder.h"
#include "Reader/CSVReader.h"

class Manager {

private:
    const std::filesystem::path databaseFolder; // database folder
    const std::filesystem::path networkFilename; // network input \ output filename

    const size_t dataLength = 28*28; // data size = image of (28 X 28) pixels

    const bool isTrueRandom = false; // whether RNG is truly random

    // Network parameters
    const size_t nEpochs = 30; // Number of epochs
    const size_t batchSize = 10; // Batch size
    const double eta = 0.1; // Learning rate
    const double lambda = 5; // Regularization parameter
    using PREDICTION = size_t; // Prediction output type

    // Determine database filenames
    const std::filesystem::path trainingFile = databaseFolder / "Training.csv";
    const std::filesystem::path validationFile = databaseFolder / "Validation.csv";
    const std::filesystem::path testingFile = databaseFolder / "Testing.csv";

    NetworkBuilder builder = NetworkBuilder(); // Network builder

    // Run-time generated
    DataLabel_Set dataTraining;
    DataLabel_Set dataValidation;
    DataLabel_Set dataTesting;
    std::shared_ptr<Network<PREDICTION>> network = nullptr;

public:
    /*
    Name: Manager
    Input: databaseFolder - Database folder path
           networkFilename - Network input \ output filename
    Purpose: Class constructor
    Notes: 
    */
    Manager(const std::filesystem::path& databaseFolder = "../Data/MNIST",
            const std::filesystem::path& networkFilename = "../network.net") 
        : databaseFolder(databaseFolder), networkFilename(networkFilename) {
    }

    /*
    Name: ~Manager
    Input: None
    Purpose: Class destructor
    Notes: 
    */
    ~Manager(void) = default;


    /*
    Name: loadDatabase
    Input: None
    Output: None
    Purpose: Loads the MNIST database
    Notes: 
    */
    void loadDatabase(void) {
        // Read input data
        std::cout << "Reading database..." << std::endl;
        auto rawDataTraining = read_csv_MNIST(trainingFile, dataLength);
        auto rawDataValidation = read_csv_MNIST(validationFile, dataLength);
        auto rawDataTesting = read_csv_MNIST(testingFile, dataLength);

        // Convert to Eigen data type
        std::cout << "Converting database to Eigen..." << std::endl;
        this->dataTraining = convert_to_eigen_set(rawDataTraining);
        this->dataValidation = convert_to_eigen_set(rawDataValidation);
        this->dataTesting = convert_to_eigen_set(rawDataTesting);

        std::cout << "Finished creating the database.\n" << std::endl;
    }

    /*
    Name: createNetwork
    Input: None
    Output: network - Neural network
    Purpose: Creates and returns an empty neural network (no layers within)
    Notes: 
    */
    std::shared_ptr<Network<PREDICTION>> createNetwork(void) {
        this->network = builder
            .setInputSize(dataLength)
            .setCostFunction(CostFunctionFactory::CostFunctions::CROSSENTROPY)
            .setOptimizer(OptimizerFactory::Optimizers::SGD)
            .setIsTrueRandom(isTrueRandom)
            .build<PREDICTION>();

        return network;
    }

    /*
    Name: loadNetwork
    Input: None
    Output: network - Neural network
    Purpose: Loads a neural network from a file
    Notes: 
    */
    std::shared_ptr<Network<PREDICTION>> loadNetwork(void) {
        this->network = builder.load<PREDICTION>(networkFilename);

        return network;
    }

    /*
    Name: saveNetwork
    Input: None
    Output: None
    Purpose: Saves a neural network to a file
    Notes: 
    */
    void saveNetwork(void) const {
        NetworkBuilder::save<PREDICTION>(network, networkFilename);
    }

    /*
    Name: trainNetwork
    Input: None
    Output: None
    Purpose: Trains the neural network according to the database
    Notes: 
    */
    void trainNetwork(void) {
        std::cout << "Training the network..." << std::endl;
        auto start = std::chrono::steady_clock::now();

        network->train(dataTraining, dataValidation, nEpochs, batchSize, eta, lambda);

        auto end = std::chrono::steady_clock::now();
        std::cout << "\nTraining has finished within "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds.\n" << std::endl;
    }

    /*
    Name: validateNetwork
    Input: index - Testing index
    Output: None
    Purpose: Tests the neural network against the testing data
    Notes: 
    */
    void validateNetwork(const size_t index = 3) const {
        std::cout << "Testing the network for test input number " << index << ":\n\t"
            << "Networks prediction is: " << network->predict(dataTesting[index].first) << ".\n\t"
            << "The actual value is: " << dataTesting[index].second(0) << ".\n" << std::endl;

        std::cout << "List of epoch accuracies for the validation set:" << std::endl;
        auto accuracies = network->evaluationAccuracy;
        for (auto&& accuracy : accuracies) {
            std::cout << accuracy << "\n";
        }

        auto [correct, cost] = network->calcAccuracyAndCost(dataTesting);
        std::cout << "\nFor the testing set: total correct = " << correct << " out of " << dataTesting.size() << std::endl;
    }
};