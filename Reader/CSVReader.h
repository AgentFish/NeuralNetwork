#pragma once

/*
FileName: CSVReader
Description: Reads a CSV file
Notes: Using CSV2 library: https://github.com/p-ranav/csv2
Author: Oren Fischman, September 2020
Edited: Oren Fischman, September 2020
*/

#include <filesystem>

#include "../libs/CSV2/csv2.h"
#include "../libs/Eigen/Core"


// Type alias
using DataLabel_Pair = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
using DataLabel_Set = std::vector<DataLabel_Pair>;

/*
Name: read_csv_MNIST
Input: filename - Name of the file to read
       splitIndex - Index where the Data ends and the Label begins
Output: result - Vector of pairs of Data & Label
Purpose: Reads MNIST file into a vector of pairs of Data & Label
Notes: 
*/
std::vector<std::pair<std::vector<double>, std::vector<double>>> read_csv_MNIST(const std::filesystem::path& filename, const size_t splitIndex) {
    // Reads a CSV file into a vector of pairs where each pair represents <data, data's label>

    // Normalize factor for image's pixel values
    const double normalizeFactor = 255;

    // Initialize output
    std::vector<std::pair<std::vector<double>, std::vector<double>>> result;

    // Generate CSV reader object
    csv2::Reader<csv2::delimiter<','>, // Delimiter = ,
                csv2::quote_character<'"'>, 
                csv2::first_row_is_header<false>, // No header row
                csv2::trim_policy::trim_whitespace> csv;

    // Read file
    if (csv.mmap(filename.u8string())) {

        // Initialize
        size_t rowIdx = 0;
        size_t colIdx;
        std::string str;

        for (const auto row : csv) {

            // Initialize a new row
            result.emplace_back(std::vector<double> {}, std::vector<double> {});
            result[rowIdx].first.reserve(splitIndex);
            // Reset column index
            colIdx = 0;

            for (const auto cell : row) {
                // Get string
                cell.read_value(str);

                // Assign string
                if (colIdx < splitIndex) {
                    // Data type
                    result[rowIdx].first.push_back(std::stod(str) / normalizeFactor);
                }
                else {
                    // Label type
                    result[rowIdx].second.push_back(std::stod(str));
                }

                // Prepare for next iteration
                str.clear();
                colIdx++; // Next column
            }

            rowIdx++; // Next row
        }
    }

    return result;
}

/*
Name: convert_to_eigen_set
Input: input - Original data
Output: result - Data in the format of Eigen
Purpose: Generate Eigen type of the databases
Notes: 
*/
DataLabel_Set convert_to_eigen_set(std::vector<std::pair<std::vector<double>, std::vector<double>>> input) {
    // Initialize output
    DataLabel_Set result;

    // Fill data
    for (size_t rowIdx = 0; rowIdx < input.size(); ++rowIdx) {
        // Initialize
        result.emplace_back(Eigen::VectorXd {}, Eigen::VectorXd {});

        // Data type
        result[rowIdx].first = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(input[rowIdx].first.data(), input[rowIdx].first.size());

        // Label type
        result[rowIdx].second = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(input[rowIdx].second.data(), input[rowIdx].second.size());
    }

    return result;
}