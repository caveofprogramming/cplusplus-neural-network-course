#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixfunctions.h"
#include "matrix.h"
#include "neuralnetwork.h"
#include "vector.h"

#include <cmath>

using namespace cave;

int main()
{
    Vector v(3);

    std::cout << v << std::endl;
    return 0;
    NeuralNetwork nn;

    const int inputSize = 4;
    const int outputSize = 5;

    nn.setScaleWeights(0.1);

    nn.add(NeuralNetwork::DENSE, 5, inputSize);
    nn.add(NeuralNetwork::RELU);
    nn.add(NeuralNetwork::DENSE, 4);
    nn.add(NeuralNetwork::RELU);
    nn.add(NeuralNetwork::DENSE, 3);
    nn.add(NeuralNetwork::SOFTMAX);
    nn.add(NeuralNetwork::CROSS_ENTROPY_LOSS);

    std::cout << nn << std::endl;

    return 0;
}
