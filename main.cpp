/*
FileName: main
Description: Testing Neural Networks
Notes: 
Author: Oren Fischman, October 2020
Edited: Oren Fischman, October 2020
*/

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include "Network/Manager.h"

int main(void)
{
    // _CrtSetBreakAlloc(197);

    try {
        // Create network manager
        Manager manager = Manager();

        auto start = std::chrono::steady_clock::now();

        // Load database
        manager.loadDatabase();

        // Create empty network
        auto network = manager.createNetwork();
        // Add network layers
        network->addLayer(NetworkBuilder::createLayer(30, ActivationFunctionFactory::ActivationFunctions::LOGISTIC))
            .addLayer(NetworkBuilder::createLayer(10, ActivationFunctionFactory::ActivationFunctions::LOGISTIC));

        // Load network from a file
        // auto network = manager.loadNetwork();

        // Print network layers
        network->printLayers();

        // Train network
        manager.trainNetwork();

        // Save network to a file
        manager.saveNetwork();

        // Test network
        manager.validateNetwork();


        auto end = std::chrono::steady_clock::now();
        std::cout << "\nTotal calculation time was "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds." << std::endl;
    }
    catch(const std::exception& exception) {
        std::cerr << exception.what() << '\n';
    }

    _CrtDumpMemoryLeaks();
    return 0;
}