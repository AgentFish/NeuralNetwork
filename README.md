## Simple C++ neural network
Simple fully connected neural network implementation in modern C++

Modular and easily upgradable.
Well suited for experimenting and learning for neural networks or C++ newcomers.
This project was developed while reading the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen.


## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Example](#example)
* [License](#license)


## Features

* Activation functions
	* Logistic
	* Softmax (unfinished)
* Cost functions
	* Quadratic
	* Cross Entropy
* Optimizers
	* Stochastic Gradient Descent
* Header only
* Linear algebra using [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library
* CSV parsing using [csv2](https://github.com/p-ranav/csv2) library


## Requirements

* C++17 and above
* CMake & VSCode are recommended


## Example
Input data set is provided as a 7-Zip file which needs to be manually extracted, to reveal the CSV files.
The data is based on [MNIST database](http://yann.lecun.com/exdb/mnist/) and split to training, validation and testing files.
An additional smaller sub-set is provided for debugging purposes (faster).

A network with 30 hidden neurons (784-30-10) completes the training session (30 epochs) with move than 96% accuracy on validation data and testing data.
Running in Release mode (optimized for speed) is recommended.


## License

The project is available under the [MIT](https://opensource.org/licenses/MIT) license.
